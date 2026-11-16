cmake_minimum_required(VERSION 3.8)
project(gazebo_ignition)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Installs launch, worlds, and config to share/gazebo_ignition
install(
  DIRECTORY launch world config map
  DESTINATION share/${PROJECT_NAME}
)

ament_package()