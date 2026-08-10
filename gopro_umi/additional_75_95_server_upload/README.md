# Additional episodes 75–95: server transfer bundle

This directory is a non-duplicating local transfer bundle for the new videos
and their processed results.

- `Episode/`: symbolic links to the original `GX010075.MP4` through
  `GX010095.MP4` files.
- `Episode_result/episode_57` through `episode_77`: symbolic links to the
  local incremental processing results.
- `episode_mapping_log.json`: mapping to merge into the server's existing
  56-episode mapping log.

Do not upload this bundle until every new episode has these files:

```text
world/trajectory_tcp_robot.csv
world/tcp_robot_validation.json  (passed: true)
gripper/gripper_width.csv
validation.json                  (passed: true)
```

Use `rsync -aL` (capital L) so symlinks are dereferenced and the server
receives real videos and result files.

Example local push after validation:

```bash
rsync -aLvhP \
  /home/kimminje/Desktop/project/gopro_umi/additional_75_95_server_upload/ \
  -e 'ssh -p 17970' \
  kimminje@155.230.189.77:/home/kimminje/gopro_umi/incoming_additional_75_95/
```

The server should merge `Episode/` and `Episode_result/episode_57..77` into
its project only after validating the transfer.  Do not replace existing
episode_1..56 or the existing map_result.
