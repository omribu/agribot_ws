#!/bin/bash

# Batch ArUco Marker Generator
# Usage: ./generate_markers.sh

echo "=== 40cm ArUco Marker Generator ==="

# Source ROS2 workspace
source ~/agribot_ws/install/setup.bash

# Calculate pixel sizes for 40cm
# 40cm = 15.75 inches
SIZE_300DPI=4724  # 15.75 * 300 = 4724 pixels (high quality)
SIZE_150DPI=2362  # 15.75 * 150 = 2362 pixels (good quality)

echo "Creating 40cm markers at 300 DPI (${SIZE_300DPI} pixels)..."
for id in {0..9}; do
    echo "Creating 40cm marker ID: $id at 300 DPI"
    ros2 run agribot_landmarks marker_creator $id $SIZE_300DPI /home/volcani/agribot_ws/src/agribot_landmarks/markers/marker_${id}_40cm_300dpi.png
done

echo "Creating 40cm markers at 150 DPI (${SIZE_150DPI} pixels)..."
for id in {0..9}; do
    echo "Creating 40cm marker ID: $id at 150 DPI"
    ros2 run agribot_landmarks marker_creator $id $SIZE_150DPI /home/volcani/agribot_ws/src/agribot_landmarks/markers/marker_${id}_40cm_150dpi.png
done

echo "=== 40cm marker generation complete! ==="
echo "Check the markers directory:"
ls -la /home/volcani/agribot_ws/src/agribot_landmarks/markers/

echo ""
echo "Printing instructions:"
echo "- For 300 DPI markers: Print at 300 DPI, no scaling"
echo "- For 150 DPI markers: Print at 150 DPI, no scaling"
echo "- Final printed size should be 40cm x 40cm"
echo ""
echo "Usage examples:"
echo "Single 40cm marker (300 DPI): ros2 run agribot_landmarks marker_creator 42 4724"
echo "Single 40cm marker (150 DPI): ros2 run agribot_landmarks marker_creator 42 2362"