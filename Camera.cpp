#include "Camera.h"
#include <iostream>

Camera::Camera() {
    // ЗАДАНИЕ ДЛЯ ВАС:
    // 1. Инициализируйте переменную cap.
    // 2. Откройте камеру с индексом 0.
    // 3. Если не открылась — выведите сообщение об ошибке в консоль.
}

Camera::~Camera() {
    // ЗАДАНИЕ ДЛЯ ВАС:
    // Освободите камеру командой cap.release();
}

cv::Mat Camera::getFrame() {
    cv::Mat frame;
    // ЗАДАНИЕ ДЛЯ ВАС:
    // 1. Считайте кадр из cap в переменную frame.
    // 2. Верните frame.
    return frame;
}