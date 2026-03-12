# Finger Math & Meme Detector (C++ / OpenCV)

[![C++](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://isocpp.org/) [![CMake](https://img.shields.io/badge/CMake-%3E%3D3.17-blueviolet.svg)](https://cmake.org/) [![OpenCV](https://img.shields.io/badge/OpenCV-%3E%3D4.0-lightgrey.svg)](https://opencv.org/) [![Platform-macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()

A small real-time computer-vision application written in modern C++ (C++23) that recognizes hand gestures from a webcam and provides two interactive modes:

- Finger Math: counts fingers on left and right hands and displays the sum.
- Meme Detector: recognizes simple gesture patterns and plays short video "memes" when gesture combos are detected.

The project uses OpenCV for camera capture, image processing, contour analysis and video playback. Unit tests are included (googletest is fetched automatically by CMake).

---

## Features

- Real-time hand segmentation (HSV-based) and contour processing.
- Palm detection using distance transform to stabilize center/radius.
- Finger counting using convex hull / convexity defects analysis with smoothing and voting.
- Two interactive UI modes (Menu / Finger Math / Meme Detector).
- Meme detection using simple heuristics over recent hand motion history (vertical/horizontal/static patterns).
- Plays bundled video clips as reactions (macOS uses `afplay` to play associated audio).

## Repository contents (important files)

- main.cpp — application entry point and main loop.
- Camera.* — simple wrapper around OpenCV VideoCapture.
- HandDetector.* — hand segmentation, face masking (uses Haar cascade), contour helpers and palm center detection.
- AppMode.* — UI modes, ROI processing, finger counting and meme-playing logic.
- MemeDetector.* — simple pattern detector for meme gestures.
- MathUtils.* — helper math utilities used by the app.
- HandState.* — smoothing / voting / history storage for each hand.
- haarcascade_frontalface_default.xml — face cascade used to suppress face regions in masks.
- 67.mov, grizman.mp4, ivanzolo.mp4 — sample meme video clips used by the Meme Detector mode.
- test_main.cpp — basic test harness; googletest is downloaded by CMake's FetchContent.
- CMakeLists.txt — build configuration.


## Requirements

- C++ compiler with C++23 support (Clang / GCC / MSVC that supports C++23).
- CMake >= 3.17
- OpenCV (>= 4.0 recommended) built with videoio and highgui.
- On macOS: recommended to install OpenCV via Homebrew: `brew install opencv`
- On Linux: install OpenCV via your distribution packages or compile from source.

Notes:
- The project CMakeLists attempts to search common prefix paths (e.g. `/opt/homebrew`, `/usr/local`). If your OpenCV installation is in a custom location, pass `-DCMAKE_PREFIX_PATH` or `-DOpenCV_DIR` to CMake.
- The app uses `afplay` on macOS to play meme audio; on Linux replace `afplay` calls with `ffplay -nodisp -autoexit` or another audio player in `AppMode.cpp` if needed.


## Quick start — macOS (Homebrew)

1. Install dependencies (example):

   brew install cmake
   brew install opencv

2. Create a build directory and run CMake:

   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/opencv@4;/opt/homebrew" ..

   # If the above OpenCV path doesn't match your installation, point OpenCV_DIR explicitly:
   # cmake -DOpenCV_DIR=/opt/homebrew/Cellar/opencv/4.x.y/lib/cmake/opencv4 ..

3. Build the project:

   cmake --build . --config Release

4. Run the application (ensure you run from the repository root or that media files and cascade are on the working directory):

   ./cv

5. Run tests:

   ./cv_tests
   # or
   ctest -V


## Quick start — Ubuntu / Debian

1. Install dependencies (approximate):

   sudo apt update
   sudo apt install build-essential cmake libopencv-dev pkg-config

2. Build and run (same as macOS):

   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . -- -j
   ./cv

If your system OpenCV package provides a non-standard CMake config path, point CMake via `-DOpenCV_DIR=/path/to/opencv4`.


## How to use

- Launch the app: it will open a window called "Finger Math App" and try to use the default camera (device 0).
- The Menu screen shows buttons: FINGER MATH, MEME DETECTOR, EXIT. Hover/press a button by placing your hand above it — the app detects a press by checking non-zero mask pixels in the button ROI.
- Finger Math: show open fingers over left and right boxes — the app counts fingers per hand and shows the sum in the center.
- Meme Detector: perform gesture combos (vertical synchronized moves, horizontal synchronized moves or static holds with specific finger counts) to trigger meme videos:
  - 67.mov — triggered by vertical synchronized motion (combo 4)
  - ivanzolo.mp4 — triggered by horizontal synchronized motion (combo 3)
  - grizman.mp4 — hold specific pose for a short duration
- Press ESC to go back to the menu or to exit (depends on mode).


## Notes & Troubleshooting

- Camera fails to open: ensure your camera is free and accessible. The app throws a CameraException on failure.
- Haar cascade: haarcascade_frontalface_default.xml is included and expected in the working directory. If you move it, update the path used by HandDetector.
- Wrong OpenCV found by CMake: pass `-DOpenCV_DIR` or `-DCMAKE_PREFIX_PATH` to CMake pointing to your OpenCV installation's `lib/cmake/opencv4` folder.
- Audio playback: macOS uses `afplay`; on Linux, modify the `system("afplay ...")` calls in AppMode.cpp to use `ffplay` or another player.
- If meme videos do not play, ensure the video files (67.mov, grizman.mp4, ivanzolo.mp4) exist in the working directory or pass full paths when creating VideoCapture in main (the current code opens these by relative name).


## Building with CLion

- Open the project in CLion; CMakeLists.txt is configured to fetch googletest for tests automatically.
- You may need to set the CMAKE_PREFIX_PATH (CLion -> Settings -> Build, Execution, Deployment -> CMake -> CMake options) to help CMake locate OpenCV on macOS (for example: `-DCMAKE_PREFIX_PATH=/opt/homebrew`).


## Running tests

- Tests are compiled into the `cv_tests` binary by the provided CMake configuration.
- From the build directory:

   ./cv_tests
   # or
   ctest -V

The project uses FetchContent to download googletest during configuration.


## Project structure (short)

- src / root files: Camera.* HandDetector.* AppMode.* MemeDetector.* MathUtils.* HandState.* main.cpp
- media: 67.mov, grizman.mp4, ivanzolo.mp4 (placed in repo root)
- resources: haarcascade_frontalface_default.xml
- tests: test_main.cpp


## Contributing

- Bug reports, suggestions and PRs welcome. Keep changes small and focused.
- If you improve the gesture detection heuristics or make audio/video playback cross-platform, please include platform-specific build/test notes.


## License

No license file is provided in the repository. Add a LICENSE file if you want to apply a specific license.


## Acknowledgements

- OpenCV (computer vision and video I/O)
- Googletest for unit testing


---

If you want, I can:
- add a sample GitHub Actions workflow for CI building and running tests,
- replace macOS-only `afplay` calls with a cross-platform approach (ffmpeg/ffplay) and update the code.


