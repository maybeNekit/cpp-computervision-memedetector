#include <iostream>
#include "Camera.h"
#include "HandDetector.h" // <--- Подключаем новый заголовок

int main() {
    Camera myCam;
    HandDetector detector; // <--- Создаем объект детектора

    while (true) {
        cv::Mat frame = myCam.getFrame();

        if (frame.empty()) break;

        // --- МАГИЯ ТУТ ---
        // Получаем черно-белую маску руки
        cv::Mat handMask = detector.detectHand(frame);

        // Показываем ДВА окна:
        cv::imshow("Original", frame);  // Обычное видео
        cv::imshow("Hand Mask", handMask); // То, как видит компьютер
        // -----------------

        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}