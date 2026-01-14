#include <iostream>
#include <cmath>
#include <numeric>
#include <deque>
#include <map>
#include "Camera.h"
#include "HandDetector.h"

using namespace cv;
using namespace std;

enum AppState {
    STATE_MENU,
    STATE_ACTIVE
};

struct HandState {
    deque<int> history;
    double smoothRadius = 0;
    Point smoothCenter = Point(0, 0);
    bool isFirstFrame = true;
    int displayedFingers = 0;
};

double getAngle(Point s, Point f, Point e) {
    double l1 = norm(f - s);
    double l2 = norm(f - e);
    double dot = (s.x - f.x) * (e.x - f.x) + (s.y - f.y) * (e.y - f.y);
    double angle = acos(dot / (l1 * l2));
    return angle * 180.0 / 3.14159265;
}

// --- ПРОВЕРКА НАЖАТИЯ (МГНОВЕННАЯ) ---
bool checkButtonPress(Mat &frame, Rect btnRect, HandDetector &detector) {
    btnRect = btnRect & Rect(0, 0, frame.cols, frame.rows);
    if (btnRect.area() == 0) return false;

    Mat roi = frame(btnRect);
    Mat mask = detector.detectHand(roi);
    int skinPixels = countNonZero(mask);

    // Порог: 60 пикселей (чуть-чуть кожи достаточно)
    return (skinPixels > 60);
}

void processHandInROI(Mat &frame, Rect roiRect, HandDetector &detector, HandState &state) {
    roiRect = roiRect & Rect(0, 0, frame.cols, frame.rows);
    if (roiRect.area() == 0) return;

    Mat roiFrame = frame(roiRect);
    Mat mask = detector.detectHand(roiFrame);
    vector<Point> rawContour = detector.findHandContour(mask);
    int currentFingers = 0;

    rectangle(frame, roiRect, Scalar(200, 200, 200), 2);

    if (!rawContour.empty()) {
        vector<Point> handContour;
        for(auto pt : rawContour) handContour.push_back(pt + roiRect.tl());

        double rawRadius = 0;
        Point localCenter = detector.getPalmCenter(mask, rawRadius);
        Point globalCenter = localCenter + roiRect.tl();

        if (state.isFirstFrame) {
            state.smoothRadius = rawRadius;
            state.smoothCenter = globalCenter;
            state.isFirstFrame = false;
        } else {
            state.smoothRadius = state.smoothRadius * 0.6 + rawRadius * 0.4;
            state.smoothCenter.x = (int)(state.smoothCenter.x * 0.6 + globalCenter.x * 0.4);
            state.smoothCenter.y = (int)(state.smoothCenter.y * 0.6 + globalCenter.y * 0.4);
        }

        circle(frame, state.smoothCenter, (int)state.smoothRadius, Scalar(0, 255, 255), 2);
        circle(frame, state.smoothCenter, 5, Scalar(0, 0, 255), -1);

        vector<int> hullIndices;
        convexHull(handContour, hullIndices, false);
        vector<Point> hullPoints;
        for(int idx : hullIndices) hullPoints.push_back(handContour[idx]);

        vector<Vec4i> defects;
        if (hullIndices.size() > 3) {
            try { convexityDefects(handContour, hullIndices, defects); } catch (...) {}
        }

        int validDefects = 0;
        for (int i = 0; i < defects.size(); i++) {
            Point p_start = handContour[defects[i][0]];
            Point p_end = handContour[defects[i][1]];
            Point p_far = handContour[defects[i][2]];
            double depth = defects[i][3] / 256.0;

            if (depth < state.smoothRadius * 0.4) continue;
            if (p_far.y > state.smoothCenter.y + state.smoothRadius) continue;

            if (getAngle(p_start, p_far, p_end) < 95) {
                validDefects++;
                circle(frame, p_far, 6, Scalar(255, 0, 0), -1);
            }
        }

        if (validDefects == 0) {
            double areaContour = contourArea(handContour);
            double areaHull = contourArea(hullPoints);
            double solidity = areaContour / areaHull;

            Point topPoint = state.smoothCenter;
            double maxDist = 0;
            int topIdx = -1;

            for (int idx : hullIndices) {
                Point pt = handContour[idx];
                if (pt.y > state.smoothCenter.y) continue;
                double d = norm(pt - state.smoothCenter);
                if (d > maxDist) { maxDist = d; topPoint = pt; topIdx = idx; }
            }

            double tipAngle = 180;
            if (topIdx != -1) {
                int sz = handContour.size();
                int offset = 12;
                if (sz > 30) {
                    Point p_prev = handContour[(topIdx - offset + sz) % sz];
                    Point p_next = handContour[(topIdx + offset) % sz];
                    tipAngle = getAngle(p_prev, topPoint, p_next);
                }
            }

            bool isLongEnough = maxDist > state.smoothRadius * 1.6;
            bool isSharp = tipAngle < 65;
            bool isNotSolid = solidity < 0.92;

            if (isLongEnough && (isSharp || isNotSolid)) {
                currentFingers = 1;
                line(frame, state.smoothCenter, topPoint, Scalar(0, 255, 0), 3);
            } else {
                currentFingers = 0;
            }
        } else {
            currentFingers = validDefects + 1;
        }
        if (currentFingers > 5) currentFingers = 5;

        vector<vector<Point>> dc = {handContour};
        drawContours(frame, dc, 0, Scalar(100, 100, 100), 1);

    } else {
        state.isFirstFrame = true;
        currentFingers = 0;
    }

    state.history.push_back(currentFingers);
    if (state.history.size() > 10) state.history.pop_front();

    map<int, int> votes;
    for (int val : state.history) votes[val]++;
    int maxVotes = 0;
    for (auto const& item : votes) {
        if (item.second > maxVotes) {
            maxVotes = item.second;
            state.displayedFingers = item.first;
        }
    }
}

// --- КЛАСС КНОПКИ (Упрощенный) ---
struct Button {
    Rect rect;
    string text;
    Scalar color;

    Button(int x, int y, int w, int h, string t) {
        rect = Rect(x, y, w, h);
        text = t;
        color = Scalar(0, 200, 0);
    }

    // Просто рисуем. Если нажата (isHovered), меняем цвет.
    void draw(Mat &frame, bool isHovered) {
        Scalar curColor = isHovered ? Scalar(0, 255, 255) : color;

        rectangle(frame, rect, curColor, 2);

        // Если нажата - заливаем полностью сразу
        if (isHovered) {
            rectangle(frame, rect, Scalar(0, 200, 0), -1);
        }

        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        int thickness = 2;
        Size textSize = getTextSize(text, fontFace, fontScale, thickness, 0);
        Point textOrg(rect.x + (rect.width - textSize.width)/2, rect.y + (rect.height + textSize.height)/2);

        Scalar textColor = isHovered ? Scalar(255, 255, 255) : curColor;
        putText(frame, text, textOrg, fontFace, fontScale, textColor, thickness);
    }
};

int main() {
    Camera myCam;
    HandDetector detector;

    HandState leftHandState;
    HandState rightHandState;
    AppState appState = STATE_MENU;

    while (true) {
        Mat frame = myCam.getFrame();
        if (frame.empty()) break;

        flip(frame, frame, 1);
        int w = frame.cols;
        int h = frame.rows;

        Button startBtn(w/2 - 150, 50, 300, 80, "FINGER MATH");
        Button exitBtn(20, 20, 100, 40, "EXIT");

        if (appState == STATE_MENU) {
            // === МЕНЮ ===
            bool isPressed = checkButtonPress(frame, startBtn.rect, detector);

            // Рисуем кнопку (она подсветится, если нажата)
            startBtn.draw(frame, isPressed);
            putText(frame, "Touch to start", Point(w/2 - 100, 160), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200,200,200), 1);

            // МГНОВЕННЫЙ ПЕРЕХОД
            if (isPressed) {
                appState = STATE_ACTIVE;
                leftHandState = HandState();
                rightHandState = HandState();
            }
        }
        else if (appState == STATE_ACTIVE) {
            // === ИГРА ===
            int boxSize = 450;
            if (h < 500) boxSize = h - 50;
            int padding = 20;

            Rect rectLeft(padding, h - boxSize - padding, boxSize, boxSize);
            Rect rectRight(w - boxSize - padding, h - boxSize - padding, boxSize, boxSize);

            processHandInROI(frame, rectLeft, detector, leftHandState);
            processHandInROI(frame, rectRight, detector, rightHandState);

            Scalar colorL = (leftHandState.displayedFingers > 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            putText(frame, "L: " + to_string(leftHandState.displayedFingers),
                    rectLeft.tl() + Point(0, -10), FONT_HERSHEY_SIMPLEX, 2, colorL, 4);

            Scalar colorR = (rightHandState.displayedFingers > 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            putText(frame, "R: " + to_string(rightHandState.displayedFingers),
                    rectRight.tl() + Point(0, -10), FONT_HERSHEY_SIMPLEX, 2, colorR, 4);

            // Кнопка Выход
            bool isExitPressed = checkButtonPress(frame, exitBtn.rect, detector);
            exitBtn.draw(frame, isExitPressed);

            // МГНОВЕННЫЙ ВЫХОД
            if (isExitPressed) {
                appState = STATE_MENU;
                // Небольшая задержка, чтобы сразу не нажать Старт снова, если рука там осталась
                waitKey(500);
            }
        }

        imshow("Finger Math App", frame);
        if (waitKey(1) == 27) break;
    }
    return 0;
}