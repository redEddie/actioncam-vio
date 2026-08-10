// Offline ORB-SLAM3 stereo-inertial driver for RealSense .db3 recordings.
//
// Reads a recorded .db3 (IR stereo + accel + gyro), replays it through an
// async callback (lossless 200 Hz IMU like the live example), and feeds
// ORB-SLAM3 in IMU_STEREO mode. A *bounded* queue gives back-pressure so the
// non-real-time playback never outruns tracking -> no dropped frames.
//
// Map atlas save/load is driven by the settings yaml
// (System.SaveAtlasToFile / System.LoadAtlasFromFile). Use --localize to put
// the system in localization mode (for processing demos against a prebuilt map).
//
//   stereo_inertial_db3 <vocab> <settings.yaml> <recording.db3> [traj_out] [--localize]
//
// Based on ORB_SLAM3 Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc

#include <signal.h>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <array>
#include <fstream>
#include <iomanip>

#include <set>
#include <opencv2/core/core.hpp>
#include <librealsense2/rs.hpp>
#include <System.h>
#include <MapPoint.h>

using namespace std;

static bool b_continue = true;
static void sigint_handler(int) { b_continue = false; }

// Linear interpolation of an accel sample onto a target (gyro) time.
static rs2_vector interp(double target, const rs2_vector &cur, double curT,
                         const rs2_vector &prev, double prevT) {
    if (prevT == 0) return cur;
    if (target > curT || !(target > prevT)) return (target > curT) ? cur : prev;
    rs2_vector v;
    double f = (target - prevT) / (curT - prevT);
    v.x = prev.x + (cur.x - prev.x) * f;
    v.y = prev.y + (cur.y - prev.y) * f;
    v.z = prev.z + (cur.z - prev.z) * f;
    return v;
}

struct Frame {
    double t;                 // seconds
    cv::Mat left, right;
    vector<ORB_SLAM3::IMU::Point> imu;
    bool eof = false;
};

int main(int argc, char **argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0]
             << " <vocab> <settings.yaml> <recording.db3> [traj_out] [--localize]\n";
        return 1;
    }
    string vocab = argv[1], settings = argv[2], bag = argv[3], traj;
    bool localize = false, use_viewer = true, stereo_only = false;
    for (int i = 4; i < argc; ++i) {
        string a = argv[i];
        if (a == "--localize") localize = true;
        else if (a == "--no-viewer") use_viewer = false;
        else if (a == "--stereo") stereo_only = true;   // non-inertial (also for D405, which has no IMU)
        else traj = a;
    }
    signal(SIGINT, sigint_handler);

    // ---- playback source ----
    rs2::config cfg;
    cfg.enable_device_from_file(bag, /*repeat=*/false);

    // ---- shared state ----
    const size_t QCAP = 30;
    queue<Frame> q;
    mutex m;
    condition_variable cv_full, cv_empty;
    bool eof = false;

    // IMU accumulation (between consecutive images)
    vector<rs2_vector> vGyro;  vector<double> vGyroT;
    rs2_vector curA{}, prevA{}; double curAT = 0, prevAT = 0;
    vector<rs2_vector> vAccelSync; vector<double> vAccelSyncT;
    double last_img_t = -1.0;

    // Unbounded, non-blocking push: real-time playback drops frames if the
    // callback blocks (it did, when the bounded queue was full), which spaces
    // out the frames SLAM sees and breaks tracking. Never block the callback;
    // let the queue absorb the backlog (a few hundred MB for a short clip).
    (void)QCAP;
    auto pushFrame = [&](Frame &&f) {
        { lock_guard<mutex> lk(m); q.push(std::move(f)); }
        cv_empty.notify_one();
    };

    auto callback = [&](const rs2::frame &frame) {
        if (rs2::frameset fs = frame.as<rs2::frameset>()) {
            double t = fs.get_timestamp() * 1e-3;
            if (last_img_t > 0 && fabs(t - last_img_t) < 1e-4) return;  // dup
            rs2::video_frame L = fs.get_infrared_frame(1);
            rs2::video_frame R = fs.get_infrared_frame(2);
            int w = L.get_width(), h = L.get_height();

            Frame f;
            f.t = t;
            f.left  = cv::Mat(cv::Size(w, h), CV_8U, (void*)L.get_data(), cv::Mat::AUTO_STEP).clone();
            f.right = cv::Mat(cv::Size(w, h), CV_8U, (void*)R.get_data(), cv::Mat::AUTO_STEP).clone();
            // finalize accel-sync up to available gyro samples
            while (vGyroT.size() > vAccelSyncT.size()) {
                size_t i = vAccelSyncT.size();
                vAccelSync.push_back(interp(vGyroT[i], curA, curAT, prevA, prevAT));
                vAccelSyncT.push_back(vGyroT[i]);
            }
            size_t n = min(vGyro.size(), vAccelSync.size());
            for (size_t i = 0; i < n; ++i)
                f.imu.emplace_back(vAccelSync[i].x, vAccelSync[i].y, vAccelSync[i].z,
                                   vGyro[i].x, vGyro[i].y, vGyro[i].z, vGyroT[i]);
            vGyro.clear(); vGyroT.clear(); vAccelSync.clear(); vAccelSyncT.clear();
            last_img_t = t;
            pushFrame(std::move(f));
        } else if (rs2::motion_frame mf = frame.as<rs2::motion_frame>()) {
            double t = mf.get_timestamp() * 1e-3;
            string sn = mf.get_profile().stream_name();
            if (sn == "Gyro") {
                vGyro.push_back(mf.get_motion_data()); vGyroT.push_back(t);
            } else if (sn == "Accel") {
                prevA = curA; prevAT = curAT;
                curA = mf.get_motion_data(); curAT = t;
            }
        }
    };

    rs2::pipeline pipe;
    rs2::pipeline_profile prof = pipe.start(cfg, callback);
    auto playback = prof.get_device().as<rs2::playback>();
    // Real-time playback: with a push (callback) pipeline, non-real-time mode
    // stalls after the first frame (it expects wait_for_frames pulls). The
    // bounded queue still gives back-pressure; on a fast host SLAM keeps up.
    playback.set_real_time(true);
    playback.set_status_changed_callback([&](rs2_playback_status s) {
        if (s == RS2_PLAYBACK_STATUS_STOPPED) {
            { lock_guard<mutex> lk(m); eof = true; }
            cv_empty.notify_all();
        }
    });

    auto sensor = stereo_only ? ORB_SLAM3::System::STEREO : ORB_SLAM3::System::IMU_STEREO;
    cout << "Offline SLAM: " << bag
         << (stereo_only ? "  [STEREO]" : "  [IMU_STEREO]")
         << (localize ? "  [localization]\n" : "  [mapping]\n");
    ORB_SLAM3::System SLAM(vocab, settings, sensor, use_viewer);
    if (localize) SLAM.ActivateLocalizationMode();
    float scale = SLAM.GetImageScale();

    size_t processed = 0;
    int idle = 0;
    std::vector<std::array<double, 8>> frame_poses;  // ts(s), tx,ty,tz, qx,qy,qz,qw (Twc, map frame)
    std::set<ORB_SLAM3::MapPoint*> seen_mps;          // accumulate tracked map points (for density viz)
    while (b_continue && !SLAM.isShutDown()) {
        Frame f;
        {
            unique_lock<mutex> lk(m);
            // Idle fallback: the playback STOPPED callback is not always
            // reliable, so two consecutive 3s waits with no frame => EOF.
            bool got = cv_empty.wait_for(lk, std::chrono::seconds(3),
                                         [&]{ return !q.empty() || eof || !b_continue; });
            if (!q.empty()) {
                idle = 0;
                f = std::move(q.front()); q.pop();
                cv_full.notify_one();
            } else {
                if (eof || !b_continue) break;
                if (!got && ++idle >= 2) { cout << "\n(no more frames -> EOF)\n"; break; }
                continue;
            }
        }
        if (!stereo_only && f.imu.empty()) continue;  // IMU_STEREO aborts on an empty IMU vector
        if (scale != 1.f) {
            cv::resize(f.left,  f.left,  cv::Size(f.left.cols * scale,  f.left.rows * scale));
            cv::resize(f.right, f.right, cv::Size(f.right.cols * scale, f.right.rows * scale));
        }
        Sophus::SE3f Tcw = SLAM.TrackStereo(f.left, f.right, f.t, f.imu);
        if (SLAM.GetTrackingState() == 2) {            // OK -> capture pose live
            Sophus::SE3f Twc = Tcw.inverse();
            Eigen::Vector3f tr = Twc.translation();
            Eigen::Quaternionf qr = Twc.unit_quaternion();
            frame_poses.push_back({f.t, tr.x(), tr.y(), tr.z(), qr.x(), qr.y(), qr.z(), qr.w()});
            for (auto *mp : SLAM.GetTrackedMapPoints())
                if (mp && !mp->isBad()) seen_mps.insert(mp);
        }
        if (++processed % 60 == 0) cout << "\r  processed " << processed << " frames" << flush;
    }
    cout << "\nFinished. frames processed = " << processed << endl;

    // Robust per-frame trajectory captured live from TrackStereo: bypasses the
    // SaveTrajectoryEuRoC multi-map empty-output bug, and in localization mode
    // gives re-localized poses against the optimized map (lower drift than mapping).
    if (!traj.empty()) {
        string framesf = traj;
        size_t d2 = framesf.rfind(".txt");
        framesf = (d2 != string::npos) ? framesf.substr(0, d2) + "_frames.txt" : framesf + "_frames";
        std::ofstream fo(framesf);
        fo << "# ts tx ty tz qx qy qz qw (Twc, map frame, live capture, seconds)\n";
        fo << std::setprecision(9) << std::fixed;
        for (auto &p : frame_poses)
            fo << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " "
               << p[4] << " " << p[5] << " " << p[6] << " " << p[7] << "\n";
        cout << "Saved live per-frame trajectory (" << frame_poses.size() << ") -> " << framesf << endl;

        // Map points actually tracked during the run (world/map frame): a quick
        // proxy for map density. Read before Shutdown (atlas still alive).
        string mpf = traj;
        size_t d3 = mpf.rfind(".txt");
        mpf = (d3 != string::npos) ? mpf.substr(0, d3) + "_mappoints.txt" : mpf + "_mappoints";
        std::ofstream mo(mpf);
        mo << "# x y z (map frame) -- tracked map points\n";
        mo << std::setprecision(6) << std::fixed;
        size_t nmp = 0;
        for (auto *mp : seen_mps) {
            if (!mp || mp->isBad()) continue;
            Eigen::Vector3f p = mp->GetWorldPos();
            mo << p.x() << " " << p.y() << " " << p.z() << "\n";
            ++nmp;
        }
        cout << "Saved tracked map points (" << nmp << ") -> " << mpf << endl;
    }

    SLAM.Shutdown();   // saves atlas if System.SaveAtlasToFile set in yaml
    if (!traj.empty()) {
        // per-frame trajectory (needed for ArUco tag calibration: match tag
        // detections to camera poses by frame timestamp). <traj>_full.txt
        string full = traj;
        size_t dot = full.rfind(".txt");
        full = (dot != string::npos) ? full.substr(0, dot) + "_full.txt" : full + "_full";
        try {
            SLAM.SaveTrajectoryEuRoC(full);
            cout << "Saved full (per-frame) trajectory -> " << full << endl;
        } catch (const std::exception &e) {
            cerr << "WARNING: full trajectory save failed (" << e.what() << ")." << endl;
        }
        try {
            SLAM.SaveKeyFrameTrajectoryEuRoC(traj);
            cout << "Saved keyframe trajectory -> " << traj << endl;
        } catch (const std::exception &e) {
            // happens when the map never properly initialized (e.g. IMU init
            // failed due to no motion) -> degenerate atlas. Report, don't abort.
            cerr << "WARNING: trajectory save failed (" << e.what()
                 << "). The map likely did not initialize (insufficient motion?)." << endl;
        }
    }
    return 0;
}
