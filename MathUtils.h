#pragma once
#include <opencv2/opencv.hpp>

constexpr double pi() { return 3.14159265358979323846; }
double getAngle(cv::Point s, cv::Point f, cv::Point e);
double getDist(cv::Point p1, cv::Point p2);