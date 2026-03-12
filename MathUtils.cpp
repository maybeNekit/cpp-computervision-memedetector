#include "MathUtils.h"
#include <cmath>

double getAngle(cv::Point s, cv::Point f, cv::Point e) {
    double l1 = cv::norm(f - s);
    double l2 = cv::norm(f - e);
    if (l1 == 0 || l2 == 0) return 0.0;
    double dot = (s.x - f.x) * (e.x - f.x) + (s.y - f.y) * (e.y - f.y);
    double angle = std::acos(dot / (l1 * l2));
    return angle * 180.0 / pi();
}

double getDist(cv::Point p1, cv::Point p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}