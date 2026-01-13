#include <iostream>
#include "Camera.h"
#include "HandDetector.h"

int main() {
    Camera myCam;
    HandDetector detector;

    while (true) {
        cv::Mat frame = myCam.getFrame();
        if (frame.empty()) break;

        // 1. Получаем маску (С УЖЕ УДАЛЕННЫМ ЛИЦОМ)
        cv::Mat mask = detector.detectHand(frame);

        // 2. Находим контур (теперь это точно рука)
        std::vector<cv::Point> handContour = detector.findHandContour(mask);

        if (!handContour.empty()) {
            // Рисуем зеленый контур руки
            std::vector<std::vector<cv::Point>> drawList;
            drawList.push_back(handContour);
            cv::drawContours(frame, drawList, -1, cv::Scalar(0, 255, 0), 2);

            // --- БОНУС: Подсчет пальцев (визуализация) ---
            std::vector<int> hullIndices;
            cv::convexHull(handContour, hullIndices, false);

            // Если рука достаточно сложная (есть пальцы)
            if (hullIndices.size() > 3) {
                std::vector<cv::Vec4i> defects;
                try {
                     cv::convexityDefects(handContour, hullIndices, defects);
                } catch (...) {}

                int fingers = 0;
                for (int i = 0; i < defects.size(); i++) {
                    // Рисуем впадины между пальцами синими точками
                    cv::Point p_far = handContour[defects[i][2]];
                    float depth = defects[i][3] / 256.0;

                    if (depth > 15) { // Только глубокие впадины
                        cv::circle(frame, p_far, 5, cv::Scalar(255, 0, 0), -1);
                        fingers++;
                    }
                }

                // Чтобы получить кол-во пальцев, обычно к впадинам прибавляют 1.
                // Но это грубая оценка.
            }
        }

        cv::imshow("Hand Tracker", frame);
        // Включи эту строку, чтобы проверить, что лицо реально черное
        cv::imshow("Mask Debug", mask);

        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}