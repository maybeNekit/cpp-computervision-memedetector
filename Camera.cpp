#include "Camera.h"
#include <iostream>


Camera::Camera() {

    cap.open(0);


    if (!cap.isOpened()) {
        std::cerr << "ОШИБКА: Не удалось открыть камеру!" << std::endl;
    } else {
        std::cout << "Камера успешно запущена!" << std::endl;
    }
}


Camera::~Camera() {

    cap.release();
    std::cout << "Камера выключена." << std::endl;
}


cv::Mat Camera::getFrame() {
    cv::Mat frame;


    cap.read(frame);

    return frame;
}