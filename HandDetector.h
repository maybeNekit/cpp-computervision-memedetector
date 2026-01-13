#pragma once
#include <opencv2/opencv.hpp>
#include <vector> // <--- ОБЯЗАТЕЛЬНО: Подключаем библиотеку для списков

class HandDetector {
public:
    HandDetector();
    ~HandDetector();

    // Старая функция: делает картинку черно-белой
    cv::Mat detectHand(cv::Mat inputFrame);

    // НОВАЯ ФУНКЦИЯ: Ищет контуры
    // Возвращает список точек (x, y), которые образуют контур руки
    std::vector<cv::Point> findLargestContour(cv::Mat mask);
};