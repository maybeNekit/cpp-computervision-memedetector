#include "HandDetector.h"
#include <iostream>

HandDetector::HandDetector() {
    std::string path = "haarcascade_frontalface_default.xml";
    if (!faceDetector.load(path)) {
        if (!faceDetector.load(path)) {
             std::cerr << "WARNING: XML файл лица не найден! Лицо не будет удалено." << std::endl;
        }
    }
}

HandDetector::~HandDetector() {}

cv::Mat HandDetector::detectHand(cv::Mat inputFrame) {
    cv::Mat hsvImage, mask;
    cv::cvtColor(inputFrame, hsvImage, cv::COLOR_BGR2HSV);

    // 1. ЦВЕТ
    // Если дырок все еще много, можно попробовать опустить нижние границы S и V
    // Например: Scalar(0, 60, 60) вместо (0, 70, 80)
    cv::Scalar lower(0, 65, 70);
    cv::Scalar upper(20, 150, 255);
    cv::inRange(hsvImage, lower, upper, mask);

    // ==========================================
    // 2. МОЩНАЯ ЧИСТКА (Fix Swiss Cheese Effect)
    // ==========================================

    // Шаг А: Убираем мелкий белый шум на фоне (точки вокруг)
    cv::Mat kernelSmall = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernelSmall);

    // Шаг Б: ГЛАВНОЕ ИСПРАВЛЕНИЕ - ЗАКРЫВАЕМ ЧЕРНЫЕ ДЫРЫ ВНУТРИ ПАЛЬЦЕВ
    // Используем ядро побольше (9x9), чтобы залить даже крупные пропуски
    cv::Mat kernelBig = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernelBig);

    // Шаг В: Еще немного расширяем, чтобы склеить разорванные фаланги
    cv::dilate(mask, mask, kernelSmall, cv::Point(-1,-1), 1);

    // Шаг Г: Размываем края, чтобы линия была плавной, а не пиксельной
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

    // ==========================================

    // 3. УДАЛЕНИЕ ЛИЦА
    std::vector<cv::Rect> faces;
    cv::Mat gray;
    cv::cvtColor(inputFrame, gray, cv::COLOR_BGR2GRAY);

    faceDetector.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(60, 60));

    for (size_t i = 0; i < faces.size(); i++) {
        cv::Rect faceRect = faces[i];
        int padding = 40;
        faceRect.x = std::max(0, faceRect.x - padding);
        faceRect.y = std::max(0, faceRect.y - padding);
        faceRect.width += padding * 2;
        faceRect.height += padding * 3;
        cv::rectangle(mask, faceRect, cv::Scalar(0), -1);
    }

    return mask;
}

std::vector<cv::Point> HandDetector::findHandContour(cv::Mat mask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> bestContour;
    double maxArea = 0;

    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea && area > 2000) {
            maxArea = area;
            bestContour = contours[i];
        }
    }
    return bestContour;
}

cv::Point HandDetector::getPalmCenter(cv::Mat mask, double &radius) {
    if (mask.empty()) return cv::Point(0, 0);

    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, 5);

    int maxIdx[2];
    double maxVal;
    cv::minMaxIdx(dist, 0, &maxVal, 0, maxIdx);

    radius = maxVal;
    return cv::Point(maxIdx[1], maxIdx[0]);
}