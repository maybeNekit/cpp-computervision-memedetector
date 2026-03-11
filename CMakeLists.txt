cmake_minimum_required(VERSION 3.17)
project(cv)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)

add_executable(cv
        main.cpp
        Camera.cpp
        Camera.h
        HandDetector.cpp
        MemeDetector.h
        MemeDetector.cpp
)

target_link_libraries(cv ${OpenCV_LIBS})