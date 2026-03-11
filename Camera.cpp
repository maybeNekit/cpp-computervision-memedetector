#include "Camera.h"

Camera::Camera() {
    cap.open(0);
    if (!cap.isOpened()) {
        throw CameraException("Failed to open camera");
    }
}

Camera::~Camera() {
    cap.release();
}

cv::Mat Camera::getFrame() {
    cv::Mat frame;
    cap.read(frame);
    return frame;
}