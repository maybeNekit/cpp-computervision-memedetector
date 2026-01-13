cmake_minimum_required(VERSION 3.17)
project(cv)

set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)

add_executable(cv
        main.cpp
        Camera.cpp
        Camera.h
        HandDetector.cpp
)

target_link_libraries(cv ${OpenCV_LIBS})