cmake_minimum_required(VERSION 3.8)
project(panda_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch meshes urdf rviz worlds models
  DESTINATION share/${PROJECT_NAME}
)

# Install scripts
install(PROGRAMS
  scripts/spawn_instruments.py
  scripts/spawn_panda_with_retry.py
  scripts/spawn_homography.py
  # scripts
  DESTINATION lib/${PROJECT_NAME}
  # DESTINATION share/${PROJECT_NAME}
)

# Install worlds
install(FILES
  worlds/test.world
  DESTINATION share/${PROJECT_NAME}/worlds
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()