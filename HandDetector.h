#pragma once
#include <opencv2/opencv.hpp>

class HandDetector {
public:
    HandDetector();
    ~HandDetector();

    // Главная функция: берет обычную картинку, возвращает черно-белую маску
    cv::Mat detectHand(cv::Mat inputFrame);
};