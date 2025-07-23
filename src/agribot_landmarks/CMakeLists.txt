cmake_minimum_required(VERSION 3.8)
project(agribot_landmarks)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)


find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(image_transport REQUIRED)





ament_python_install_package(${PROJECT_NAME})



# # For CPP #
# add_executable(aruco_detector src/aruco_detector.cpp)
# ament_target_dependencies(aruco_detector 
#   rclcpp 
#   std_msgs 
#   geometry_msgs 
#   sensor_msgs 
#   visualization_msgs 
#   tf2 
#   tf2_ros 
#   tf2_geometry_msgs
#   std_msgs
#   image_transport
#   cv_bridge
# )
# target_link_libraries(aruco_detector ${OpenCV_LIBS})
# target_include_directories(aruco_detector PRIVATE ${OpenCV_INCLUDE_DIRS})

# add_executable(marker_creator src/marker_creator.cpp)
# target_link_libraries(marker_creator ${OpenCV_LIBS})
# target_include_directories(marker_creator PRIVATE ${OpenCV_INCLUDE_DIRS})

# install(TARGETS
# marker_creator aruco_detector
#   DESTINATION lib/${PROJECT_NAME}
# )

# For python #
install(PROGRAMS
  ${PROJECT_NAME}/aruco_detector.py
  ${PROJECT_NAME}/aruco_creator.py
  DESTINATION lib/${PROJECT_NAME}
)


install(
  DIRECTORY config launch markers
  DESTINATION share/${PROJECT_NAME}
)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
