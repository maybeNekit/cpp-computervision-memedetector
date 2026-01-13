#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class HandDetector {
private:
    cv::CascadeClassifier faceDetector;

public:
    HandDetector();
    ~HandDetector();

    cv::Mat detectHand(cv::Mat inputFrame);
    std::vector<cv::Point> findHandContour(cv::Mat mask);

    cv::Point getPalmCenter(cv::Mat mask, double &radius);
};