#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Trick to access private/protected members of System and Atlas
#define private public
#define protected public
#include "System.h"
#include "Atlas.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#undef private
#undef protected

#include "Converter.h"

using namespace std;
using namespace ORB_SLAM3;

int main(int argc, char **argv) {
    if(argc < 5) {
        cerr << "Usage: ./export_map -s settings.yaml -l map_atlas.osa -o map_points.ply [-k keyframes.csv]" << endl;
        return 1;
    }

    string settings_path;
    string load_path;
    string out_ply;
    string out_kf;

    for(int i = 1; i < argc; i++) {
        string arg = argv[i];
        if(arg == "-s" && i+1 < argc) settings_path = argv[++i];
        else if(arg == "-l" && i+1 < argc) load_path = argv[++i];
        else if(arg == "-o" && i+1 < argc) out_ply = argv[++i];
        else if(arg == "-k" && i+1 < argc) out_kf = argv[++i];
    }

    if(load_path.empty() || out_ply.empty()) {
        cerr << "Missing required arguments (-l, -o)" << endl;
        return 1;
    }

    cout << "Loading map from: " << load_path << endl;

    string vocab_path = "/ORB_SLAM3/Vocabulary/ORBvoc.txt";

    // Initialize System with map loading
    System SLAM(vocab_path, settings_path, System::IMU_MONOCULAR, false, load_path);

    cout << "Map loaded! Extracting points..." << endl;

    vector<Map*> vpMaps = SLAM.mpAtlas->GetAllMaps();
    int nPoints = 0;

    for(Map* pMap : vpMaps) {
        vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
        for(MapPoint* pMP : vpMPs) {
            if(!pMP || pMP->isBad()) continue;
            nPoints++;
        }
    }

    cout << "Found " << nPoints << " active map points." << endl;

    // Export PLY
    ofstream f(out_ply);
    f << "ply\nformat ascii 1.0\n";
    f << "element vertex " << nPoints << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property int map_id\n";
    f << "end_header\n";

    for(Map* pMap : vpMaps) {
        int map_id = pMap->GetId();
        vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
        for(MapPoint* pMP : vpMPs) {
            if(!pMP || pMP->isBad()) continue;
            Eigen::Vector3f p = pMP->GetWorldPos();
            f << fixed << p(0) << " " << p(1) << " " << p(2) << " " << map_id << "\n";
        }
    }
    f.close();
    cout << "Saved map points to " << out_ply << endl;

    if(!out_kf.empty()) {
        ofstream f_kf(out_kf);
        f_kf << "timestamp,x,y,z,q_w,q_x,q_y,q_z,map_id\n";
        int nKFs = 0;
        for(Map* pMap : vpMaps) {
            int map_id = pMap->GetId();
            vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
            for(KeyFrame* pKF : vpKFs) {
                if(!pKF || pKF->isBad()) continue;
                Sophus::SE3f Twc = pKF->GetPoseInverse();
                Eigen::Vector3f twc = Twc.translation();
                Eigen::Quaternionf q = Twc.unit_quaternion();
                f_kf << fixed << pKF->mTimeStamp << ","
                     << twc(0) << "," << twc(1) << "," << twc(2) << ","
                     << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "," << map_id << "\n";
                nKFs++;
            }
        }
        f_kf.close();
        cout << "Saved " << nKFs << " keyframes to " << out_kf << endl;
    }

    return 0;
}
