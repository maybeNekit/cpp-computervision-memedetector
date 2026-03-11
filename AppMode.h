#pragma once
#include <opencv2/opencv.hpp>
#include "HandDetector.h"

enum class NextState {
    KEEP_CURRENT,
    GO_MENU,
    GO_MATH,
    GO_MEME,
    EXIT
};

class AppMode {
public:
    virtual ~AppMode() = default; 
    virtual NextState update(cv::Mat& frame, cv::Mat& globalMask, HandDetector& detector) = 0;
};
