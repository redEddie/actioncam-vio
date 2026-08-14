# Additional episodes 75–95: transfer provenance

The temporary root-level `additional_75_95_server_upload/` bundle was a
non-duplicating set of symlinks used for the completed server handoff. It was
removed during canonical cleanup on 2026-08-14 because it contained no data:
21 video links plus 21 result links, of which the result links still pointed
to the obsolete pre-migration `1_data_pipeline/actioncam-vio` path.

Canonical physical ownership remains:

- videos/results: `1_capture/actioncam-vio/Episode/` and
  `1_capture/actioncam-vio/Episode_result/`;
- non-duplicating project views: `1_capture/recordings/gopro/Episode` and
  `2_dataset/source/actioncam_vio/episode_results`;
- transfer mapping provenance: this directory's `episode_mapping_log.json`.

Do not upload this bundle until every new episode has these files:

```text
world/trajectory_tcp_robot.csv
world/tcp_robot_validation.json  (passed: true)
gripper/gripper_width.csv
validation.json                  (passed: true)
```

If another transfer bundle is ever required, recreate it from the canonical
actioncam repository and verify every target before using `rsync -aL`. Do not
restore or depend on the obsolete root-level staging path.
