#pragma once
#include <opencv2/opencv.hpp>

class Camera {
private:

    cv::VideoCapture cap;

public:

    Camera();


    ~Camera();

    cv::Mat getFrame(); 
};