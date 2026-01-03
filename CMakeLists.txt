cmake_minimum_required(VERSION 3.17)
project(cv)

set(CMAKE_CXX_STANDARD 14)

# 1. Находим OpenCV
find_package(OpenCV REQUIRED)

# 2. Подключаем пути к заголовкам
include_directories(${OpenCV_INCLUDE_DIRS})

# 3. Собираем проект
# ВАЖНО: Тут мы убрали приставку "src/", так как файлы лежат рядом
add_executable(cv
        main.cpp
        Camera.cpp
        Camera.h
)

# 4. Линкуем библиотеки
target_link_libraries(cv ${OpenCV_LIBS})