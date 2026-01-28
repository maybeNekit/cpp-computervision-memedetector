#include "MemeDetector.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <vector>

MemeType MemeDetector::detect(const std::deque<cv::Point>& history, int fingers) {
    if (history.size() < 5) return MEME_NONE;

    std::vector<int> x_vals;
    std::vector<int> y_vals;

    for (const auto& p : history) {
        x_vals.push_back(p.x);
        y_vals.push_back(p.y);
    }

    auto minmax_x = std::minmax_element(x_vals.begin(), x_vals.end());
    auto minmax_y = std::minmax_element(y_vals.begin(), y_vals.end());

    double deltaX = *minmax_x.second - *minmax_x.first;
    double deltaY = *minmax_y.second - *minmax_y.first;

    bool isVerticalMotion = deltaY > deltaX * 1.2;
    bool isSignificant = deltaY > 50;

    if (isVerticalMotion && isSignificant) {
        return MEME_67;
    }

    return MEME_NONE;
}