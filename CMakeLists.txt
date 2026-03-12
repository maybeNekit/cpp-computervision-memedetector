cmake_minimum_required(VERSION 3.17)
project(cv)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)

add_library(cv_core
        Camera.cpp
        Camera.h
        HandDetector.cpp
        HandDetector.h
        MemeDetector.cpp
        MemeDetector.h
        AppMode.h
)

target_link_libraries(cv_core PUBLIC ${OpenCV_LIBS})

add_executable(cv main.cpp)
target_link_libraries(cv PRIVATE cv_core)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(cv_tests test_main.cpp)
target_link_libraries(cv_tests PRIVATE cv_core gtest_main)

include(GoogleTest)
gtest_discover_tests(cv_tests)