#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class HandDetector {
private:
    cv::CascadeClassifier faceDetector; // Детектор лиц

public:
    HandDetector(); // Тут будем загружать детектор
    ~HandDetector();

    // Эта функция теперь будет сама удалять лицо
    cv::Mat detectHand(cv::Mat inputFrame);

    // А эта будет просто искать самый большой объект (руку)
    std::vector<cv::Point> findHandContour(cv::Mat mask);
};