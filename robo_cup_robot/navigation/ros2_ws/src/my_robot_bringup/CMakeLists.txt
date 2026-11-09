cmake_minimum_required(VERSION 3.8)
project(my_robot_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config folder if you have configuration/yaml files in it
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  launch/lidar_filter.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()