cmake_minimum_required(VERSION 3.16)
project(udp_frame_saver_example LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH "/usr")
set(CMAKE_CXX_STANDARD 17 REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

find_package(MAVSDK REQUIRED)
find_package(CURL REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)


add_executable(take_image
  mission.cpp
)



target_include_directories(take_image PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(take_image PUBLIC ${GST_INCLUDE_DIRS})
target_include_directories(take_image PUBLIC ${JSONCPP_INCLUDE_DIRS})

target_link_libraries(take_image PUBLIC
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
    MAVSDK::mavsdk
    CURL::libcurl
)

target_compile_options(take_image PUBLIC ${GST_CFLAGS_OTHER})
