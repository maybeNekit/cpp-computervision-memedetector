#include <iostream>
#include <cmath>
#include <numeric>
#include <deque>
#include <map>
#include "Camera.h"
#include "HandDetector.h"

using namespace cv;
using namespace std;

double getAngle(Point s, Point f, Point e) {
    double l1 = norm(f - s);
    double l2 = norm(f - e);
    double dot = (s.x - f.x) * (e.x - f.x) + (s.y - f.y) * (e.y - f.y);
    double angle = acos(dot / (l1 * l2));
    return angle * 180.0 / 3.14159265;
}

double getDist(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

int main() {
    Camera myCam;
    HandDetector detector;

    std::deque<int> fingerHistory;
    const int bufferSize = 10;
    int displayedFingers = 0;

    double avgPalmRadius = 0;
    double avgMaxDist = 0;
    bool isFirstFrame = true;

    while (true) {
        Mat frame = myCam.getFrame();
        if (frame.empty()) break;

        flip(frame, frame, 1);

        Mat mask = detector.detectHand(frame);

        // Морфология
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 1. Находим сырой контур
        vector<Point> rawContour = detector.findHandContour(mask);

        int currentFrameFingers = 0;

        // Объявляем handContour здесь, чтобы он был доступен, если понадобится
        vector<Point> handContour;

        if (!rawContour.empty()) {

            // СГЛАЖИВАНИЕ (УТЮГ)
            approxPolyDP(rawContour, handContour, 3.0, true);

            double rawPalmRadius = 0;
            Point palmCenter = detector.getPalmCenter(mask, rawPalmRadius);

            if (isFirstFrame) {
                avgPalmRadius = rawPalmRadius;
                isFirstFrame = false;
            } else {
                avgPalmRadius = (avgPalmRadius * 0.6) + (rawPalmRadius * 0.4);
            }

            // Рисуем ладонь
            circle(frame, palmCenter, (int)avgPalmRadius, Scalar(0, 255, 255), 2);

            vector<int> hullIndices;
            convexHull(handContour, hullIndices, false);

            vector<Point> hullPoints;
            for(int idx : hullIndices) hullPoints.push_back(handContour[idx]);

            vector<Vec4i> defects;
            if (hullIndices.size() > 3) {
                try {
                    convexityDefects(handContour, hullIndices, defects);
                } catch (...) {}
            }

            // Адаптивный круг
            double rawMaxDist = 0;
            for (const auto& pt : hullPoints) {
                double d = getDist(pt, palmCenter);
                if (d > rawMaxDist) rawMaxDist = d;
            }
            if (avgMaxDist == 0) avgMaxDist = rawMaxDist;
            else avgMaxDist = (avgMaxDist * 0.6) + (rawMaxDist * 0.4);

            double fingerRegion = avgMaxDist - avgPalmRadius;
            double cutoffRadius = avgPalmRadius + (fingerRegion * 0.5);
            cutoffRadius = std::max(cutoffRadius, avgPalmRadius * 1.3);

            circle(frame, palmCenter, (int)cutoffRadius, Scalar(255, 0, 0), 1);

            // БЛОК 1: ПОДСЧЕТ ВПАДИН
            vector<Point> validGaps;

            for (size_t i = 0; i < defects.size(); i++) {
                Point p_start = handContour[defects[i][0]];
                Point p_end = handContour[defects[i][1]];
                Point p_far = handContour[defects[i][2]];
                double depth = defects[i][3] / 256.0;

                if (depth < avgPalmRadius * 0.35) continue;
                if (p_far.y > palmCenter.y + avgPalmRadius) continue;

                double distStart = getDist(p_start, palmCenter);
                double distEnd = getDist(p_end, palmCenter);
                if (distStart < cutoffRadius || distEnd < cutoffRadius) continue;

                double angle = getAngle(p_start, p_far, p_end);
                double tipDistance = getDist(p_start, p_end);

                bool isStandardGap = (angle < 95);
                bool isThumbGap = (angle >= 95 && angle < 125 && tipDistance > avgPalmRadius);

                if (isStandardGap || isThumbGap) {
                    bool isDuplicate = false;
                    for(const auto& existingGap : validGaps) {
                        if (getDist(p_far, existingGap) < 20) {
                            isDuplicate = true;
                            break;
                        }
                    }

                    if (!isDuplicate) {
                        validGaps.push_back(p_far);
                        circle(frame, p_far, 6, Scalar(0, 255, 0), -1);
                        line(frame, p_start, p_far, Scalar(0, 255, 0), 1);
                        line(frame, p_end, p_far, Scalar(0, 255, 0), 1);
                    }
                }
            }

            int defectsCount = validGaps.size();

            // БЛОК 2: КУЛАК ИЛИ 1 ПАЛЕЦ
            if (defectsCount == 0) {
                double areaContour = contourArea(handContour);
                double areaHull = contourArea(hullPoints);
                double solidity = areaContour / areaHull;

                Point topPoint = palmCenter;
                double maxD = 0;
                for (int idx : hullIndices) {
                    Point pt = handContour[idx];
                    if (pt.y > palmCenter.y) continue;
                    double d = norm(pt - palmCenter);
                    if (d > maxD) { maxD = d; topPoint = pt; }
                }

                if (solidity < 0.95 && maxD > cutoffRadius) {
                     currentFrameFingers = 1;
                     line(frame, palmCenter, topPoint, Scalar(0, 255, 0), 2);
                } else {
                     currentFrameFingers = 0;
                }
            }
            else {
                currentFrameFingers = defectsCount + 1;
            }

            if (currentFrameFingers > 5) currentFrameFingers = 5;

            // Отрисовка
            vector<vector<Point>> contours = {handContour};
            drawContours(frame, contours, 0, Scalar(100, 100, 100), 1);
        }
        else {
            isFirstFrame = true;
            currentFrameFingers = 0;
        }

        // БЛОК 3: ГОЛОСОВАНИЕ (Исправлено под C++14)
        fingerHistory.push_back(currentFrameFingers);
        if (fingerHistory.size() > bufferSize) fingerHistory.pop_front();

        map<int, int> votes;
        for (int f : fingerHistory) votes[f]++;

        int maxVotes = 0;
        int winner = 0;

        // Исправленный цикл for для C++14 (без structured binding)
        for (auto const& item : votes) {
            int fingerCount = item.first;
            int voteCount = item.second;

            if (voteCount > maxVotes) {
                maxVotes = voteCount;
                winner = fingerCount;
            }
        }

        int threshold = (int)(bufferSize * 0.6);
        if (maxVotes > threshold) {
            displayedFingers = winner;
        }

        // Исправленная проверка на пустую руку (используем rawContour)
        if (rawContour.empty()) displayedFingers = 0;

        string info = "Fingers: " + to_string(displayedFingers);
        Scalar textColor = (displayedFingers > 0) ? ((maxVotes > 8) ? Scalar(0, 255, 0) : Scalar(0, 255, 255)) : Scalar(0, 0, 255);
        putText(frame, info, Point(30, 50), FONT_HERSHEY_SIMPLEX, 1.5, textColor, 3);

        imshow("Hand Tracker (Final Fix)", frame);
        if (waitKey(1) == 27) break;
    }
    return 0;
}