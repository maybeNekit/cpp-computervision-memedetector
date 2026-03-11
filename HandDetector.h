#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <optional>

enum MemeType {
    MEME_67,
    MEME_GRIZMAN
};



class MemeDetector {
public:
    std::optional<MemeType> detect(const std::deque<cv::Point>& positionHistory, int fingerCount);
};
