#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

enum MemeType {
    MEME_NONE,
    MEME_67
};

class MemeDetector {
public:
    MemeType detect(const std::deque<cv::Point>& positionHistory, int fingerCount);
};