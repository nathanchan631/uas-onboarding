
cmake_minimum_required(VERSION 3.16)
project(udp_frame_saver_example LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH "/usr")
set(CMAKE_CXX_STANDARD 17 REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(MAVSDK REQUIRED)
pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0)

add_executable(take_image
  mission.cpp
  camera.cpp
  detector.cpp)

target_include_directories(take_image PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(take_image PUBLIC ${GST_INCLUDE_DIRS})
target_link_libraries(take_image PUBLIC
    ${GST_LIBRARIES}
    ${OpenCV_LIBS}
    MAVSDK::mavsdk
)

target_compile_options(take_image PUBLIC ${GST_CFLAGS_OTHER})
