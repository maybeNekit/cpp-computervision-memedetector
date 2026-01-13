#include <iostream>
#include "Camera.h"
#include "HandDetector.h"

int main() {
    Camera myCam;
    HandDetector detector;

    while (true) {
        cv::Mat frame = myCam.getFrame();
        if (frame.empty()) break;

        // 1. Получаем маску (Белые пятна)
        cv::Mat mask = detector.detectHand(frame);

        // 2. Находим контур (Математическую фигуру руки)
        std::vector<cv::Point> handContour = detector.findLargestContour(mask);

        // 3. Рисуем контур (если рука найдена)
        if (!handContour.empty()) {
            // Функция рисования требует список контуров, поэтому кладем нашу руку в список
            std::vector<std::vector<cv::Point>> contoursToDraw;
            contoursToDraw.push_back(handContour);

            // Рисуем зеленую линию толщиной 3
            // frame — холст
            // contoursToDraw — что рисовать
            // -1 — значит "рисовать всё из списка"
            // cv::Scalar(0, 255, 0) — Зеленый цвет (Green)
            cv::drawContours(frame, contoursToDraw, -1, cv::Scalar(0, 255, 0), 3);
        }

        // Показываем результат
        cv::imshow("Hand Tracker", frame);

        // Можете раскомментировать следующую строку, чтобы видеть и маску тоже
        // cv::imshow("Mask", mask);

        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}