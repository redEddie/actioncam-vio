#!/bin/bash
set -e
cd /ORB_SLAM3/build
echo "Compiling export_map.cc..."
CXX_DEFINES=$(grep "CXX_DEFINES =" CMakeFiles/gopro_slam.dir/flags.make | cut -d "=" -f 2-)
CXX_INCLUDES=$(grep "CXX_INCLUDES =" CMakeFiles/gopro_slam.dir/flags.make | cut -d "=" -f 2-)
CXX_FLAGS=$(grep "CXX_FLAGS =" CMakeFiles/gopro_slam.dir/flags.make | cut -d "=" -f 2-)
/usr/bin/c++ $CXX_DEFINES $CXX_INCLUDES $CXX_FLAGS -c /ORB_SLAM3/Examples/Monocular-Inertial/export_map.cc -o export_map.o

echo "Linking export_map..."
LINK_CMD=$(cat CMakeFiles/gopro_slam.dir/link.txt)
LINK_CMD=${LINK_CMD/CMakeFiles\/gopro_slam.dir\/Examples\/Monocular-Inertial\/gopro_slam.cc.o/export_map.o}
LINK_CMD=${LINK_CMD/-o ..\/Examples\/Monocular-Inertial\/gopro_slam/-o ..\/Examples\/Monocular-Inertial\/export_map}
eval $LINK_CMD
echo "Done!"
