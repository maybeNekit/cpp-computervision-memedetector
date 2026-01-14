#include <iostream>
#include <cmath>
#include <numeric>
#include "Camera.h"
#include "HandDetector.h"

// Функция расчета угла
double getAngle(cv::Point s, cv::Point f, cv::Point e) {
    double l1 = cv::norm(f - s);
    double l2 = cv::norm(f - e);
    double dot = (s.x - f.x) * (e.x - f.x) + (s.y - f.y) * (e.y - f.y);
    double angle = std::acos(dot / (l1 * l2));
    return angle * 180.0 / 3.14159265;
}

int main() {
    Camera myCam;
    HandDetector detector;

    while (true) {
        cv::Mat frame = myCam.getFrame();
        if (frame.empty()) break;

        cv::Mat mask = detector.detectHand(frame);
        std::vector<cv::Point> handContour = detector.findHandContour(mask);

        if (!handContour.empty()) {
            double palmRadius = 0;
            cv::Point palmCenter = detector.getPalmCenter(mask, palmRadius);

            // Визуализация тела ладони
            cv::circle(frame, palmCenter, (int)palmRadius, cv::Scalar(0, 255, 255), 2);
            cv::circle(frame, palmCenter, 5, cv::Scalar(0, 0, 255), -1);

            // Convex Hull (индексы и точки)
            std::vector<int> hullIndices;
            cv::convexHull(handContour, hullIndices, false);

            // Нам нужны точки оболочки для расчета площади
            std::vector<cv::Point> hullPoints;
            for(int idx : hullIndices) hullPoints.push_back(handContour[idx]);

            // Defects
            std::vector<cv::Vec4i> defects;
            if (hullIndices.size() > 3) {
                try {
                    cv::convexityDefects(handContour, hullIndices, defects);
                } catch (...) {}
            }

            int validDefects = 0;

            // --- 1. СЧИТАЕМ ПАЛЬЦЫ (если их 2 и больше) ---
            for (int i = 0; i < defects.size(); i++) {
                cv::Point p_start = handContour[defects[i][0]];
                cv::Point p_end = handContour[defects[i][1]];
                cv::Point p_far = handContour[defects[i][2]];
                double depth = defects[i][3] / 256.0;

                if (depth < palmRadius * 0.4) continue;
                if (p_far.y > palmCenter.y + palmRadius) continue; // Фильтр низа

                double angle = getAngle(p_start, p_far, p_end);
                if (angle < 95) {
                    validDefects++;
                    // Рисуем впадины
                    cv::circle(frame, p_far, 6, cv::Scalar(255, 0, 0), -1);
                }
            }

            int fingers = 0;

            // --- 2. БИТВА: КУЛАК (0) ПРОТИВ ПАЛЬЦА (1) ---
            if (validDefects == 0) {

                // А. Проверка Плотности (Solidity)
                // Кулак занимает почти всю площадь своей оболочки (Solidity ~ 0.95)
                // Палец торчит, создавая пустоты (Solidity < 0.9)
                double areaContour = cv::contourArea(handContour);
                double areaHull = cv::contourArea(hullPoints);
                double solidity = areaContour / areaHull;

                // Б. Поиск самой высокой точки (минимальный Y)
                // Мы ищем не просто "далекую", а именно "верхнюю" точку
                cv::Point topPoint = palmCenter;
                double maxDist = 0;
                int topIdx = -1;

                for (int idx : hullIndices) {
                    cv::Point pt = handContour[idx];

                    // Строгий фильтр: Игнорируем всё, что ниже середины ладони
                    if (pt.y > palmCenter.y) continue;

                    double d = cv::norm(pt - palmCenter);
                    if (d > maxDist) {
                        maxDist = d;
                        topPoint = pt;
                        topIdx = idx;
                    }
                }

                // В. Расчет угла на кончике (ОСТРОТА)
                double tipAngle = 180; // По умолчанию тупой
                if (topIdx != -1) {
                    int sz = handContour.size();
                    int offset = 12; // Смотрим узкий кончик
                    if (sz > 30) {
                        cv::Point p_prev = handContour[(topIdx - offset + sz) % sz];
                        cv::Point p_next = handContour[(topIdx + offset) % sz];
                        tipAngle = getAngle(p_prev, topPoint, p_next);
                    }
                }

                // --- ЛОГИКА РЕШЕНИЯ ---
                // Чтобы стать пальцем, нужно пройти ВСЕ проверки:

                bool isLongEnough = maxDist > palmRadius * 1.6;  // 1. Длина
                bool isSharp = tipAngle < 65;                    // 2. Острота (Костяшка тупая ~100, Палец острый ~40)
                bool isNotSolid = solidity < 0.92;               // 3. Не кулак (Кулак очень плотный)

                // ДЕБАГ: Рисуем линию к кандидату
                if (topIdx != -1) {
                   cv::line(frame, palmCenter, topPoint, cv::Scalar(100, 100, 100), 1);
                }

                // ЕСЛИ это достаточно длинно И (остро ИЛИ не плотно) -> Палец
                // Мы разрешаем "не плотно" быть заменой "остро", если палец немного кривой
                if (isLongEnough && (isSharp || isNotSolid)) {
                    fingers = 1;
                    // Подтверждаем зеленым цветом
                    cv::line(frame, palmCenter, topPoint, cv::Scalar(0, 255, 0), 3);
                } else {
                    fingers = 0; // Кулак
                }

                // Вывод параметров для настройки (чтобы ты видел цифры)
                std::string debugInfo = "Solidity: " + std::to_string(solidity).substr(0,4) +
                                      " Angle: " + std::to_string((int)tipAngle);
                cv::putText(frame, debugInfo, cv::Point(10, frame.rows - 20),
                           cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 255), 1);

            } else {
                fingers = validDefects + 1;
            }

            std::string info = "Fingers: " + std::to_string(fingers);
            cv::putText(frame, info, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 3);

            std::vector<std::vector<cv::Point>> contours = {handContour};
            cv::drawContours(frame, contours, 0, cv::Scalar(100, 100, 100), 1);
        }

        cv::imshow("Finger Counter", frame);
        cv::imshow("Mask Debug", mask);

        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}