#pragma once
#include <opencv2/opencv.hpp>
#include <deque>

struct HandState {
    std::deque<int> fingerHistory;
    std::deque<double> radiusBuffer;
    std::deque<cv::Point> positionHistory;
    static const int bufferLimit = 100;

    cv::Point smoothCenter = cv::Point(0, 0);
    double smoothRadius = 0;
    bool isFirstFrame = true;
    int displayedFingers = 0;

    void reset();
    void updateVoting(int currentFingers);
};