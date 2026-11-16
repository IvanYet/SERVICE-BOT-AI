cmake_minimum_required(VERSION 3.8)
project(arm_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

# Install your directories to the share folder so ROS can find them at runtime
install(DIRECTORY
  rviz
  launch
  urdf
  mesh
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
