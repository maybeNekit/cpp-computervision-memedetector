#include "HandDetector.h"
#include <iostream>

HandDetector::HandDetector() {
    std::string path = "haarcascade_frontalface_default.xml";
    if (!faceDetector.load(path)) {
         std::cerr << "WARNING: XML файл лица не найден!" << std::endl;
    }
}

HandDetector::~HandDetector() {}

cv::Mat HandDetector::detectHand(cv::Mat inputFrame) {
    cv::Mat hsvImage, mask;
    cv::cvtColor(inputFrame, hsvImage, cv::COLOR_BGR2HSV);

    // 1. ЦВЕТ
    cv::Scalar lower(0, 60, 60);
    cv::Scalar upper(20, 150, 255);
    cv::inRange(hsvImage, lower, upper, mask);

    // 2. ЧИСТКА
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::dilate(mask, mask, kernel, cv::Point(-1,-1), 2);
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

    // 3. УДАЛЕНИЕ ЛИЦА
    std::vector<cv::Rect> faces;
    cv::Mat gray;
    cv::cvtColor(inputFrame, gray, cv::COLOR_BGR2GRAY);
    faceDetector.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(60, 60));

    for (size_t i = 0; i < faces.size(); i++) {
        cv::Rect r = faces[i];
        r.x = std::max(0, r.x - 40);
        r.y = std::max(0, r.y - 40);
        r.width += 80;
        r.height += 200;
        cv::rectangle(mask, r, cv::Scalar(0), -1);
    }

    return mask;
}

std::vector<cv::Point> HandDetector::findHandContour(cv::Mat mask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> bestContour;
    double maxArea = 0;

    for (size_t i = 0; i < contours.size(); i++) {
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


    cv::Rect boundRect = cv::boundingRect(mask);
    if (boundRect.area() == 0) return cv::Point(0, 0);

    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, 5);



    int limitHeight = std::min(boundRect.height, (int)(boundRect.width * 1.2));

    cv::Rect searchZone = boundRect;
    searchZone.height = limitHeight;


    searchZone = searchZone & cv::Rect(0, 0, mask.cols, mask.rows);

    double maxVal;
    cv::Point maxLoc;
    cv::Mat distROI = dist(searchZone);

    cv::minMaxLoc(distROI, 0, &maxVal, 0, &maxLoc);


    maxLoc.x += searchZone.x;
    maxLoc.y += searchZone.y;

    radius = maxVal;
    return maxLoc;
}