cmake_minimum_required(VERSION 3.17)
project(cv)

set(CMAKE_CXX_STANDARD 14)

# --- МАГИЯ УНИВЕРСАЛЬНОСТИ ---
# Мы говорим CMake: "Поищи библиотеки в этих стандартных местах Homebrew"
# Это покрывает и Intel Mac (/usr/local), и Apple Silicon (/opt/homebrew)
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")

# Теперь ищем OpenCV. Если он установлен через brew, он найдется.
find_package(OpenCV REQUIRED)

# Подключаем заголовки
include_directories(${OpenCV_INCLUDE_DIRS})
# Подключаем текущую папку (для Camera.h)
include_directories(.)

# Собираем проект
add_executable(cv
        main.cpp
        Camera.cpp
        Camera.h
)

# Линкуем библиотеки
target_link_libraries(cv ${OpenCV_LIBS})