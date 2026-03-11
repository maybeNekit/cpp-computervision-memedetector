#pragma once
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

class CameraException : public std::runtime_error {
public:
    explicit CameraException(const std::string& msg) : std::runtime_error(msg) {}
};

class Camera {
private:
    cv::VideoCapture cap;


public:
    Camera();
    ~Camera();
    cv::Mat getFrame(); 
};
