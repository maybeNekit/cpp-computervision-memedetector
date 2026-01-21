#include <iostream>
#include <cmath>
#include <numeric>
#include <deque>
#include <vector>
#include <algorithm>
#include <map>
#include "Camera.h"
#include "HandDetector.h"
#include "MemeDetector.h"

using namespace cv;
using namespace std;

enum AppState {
    STATE_START_SCREEN,
    STATE_MATH,
    STATE_MEME
};

struct HandState {
    deque<int> fingerHistory;
    deque<double> radiusBuffer;
    deque<Point> positionHistory;
    static const int bufferLimit = 100;

    Point smoothCenter = Point(0, 0);
    double smoothRadius = 0;
    bool isFirstFrame = true;

    int displayedFingers = 0;

    void reset() {
        fingerHistory.clear();
        radiusBuffer.clear();
        positionHistory.clear();
        smoothCenter = Point(0, 0);
        smoothRadius = 0;
        isFirstFrame = true;
        displayedFingers = 0;
    }
};

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

bool checkButtonPress(Mat &frame, Rect btnRect, HandDetector &detector) {
    btnRect = btnRect & Rect(0, 0, frame.cols, frame.rows);
    if (btnRect.area() == 0) return false;
    Mat roi = frame(btnRect);
    Mat mask = detector.detectHand(roi);
    int skinPixels = countNonZero(mask);
    return (skinPixels > 60);
}

void processHandInROI(Mat &frame, Rect roiRect, Mat &maskROI, HandDetector &detector, HandState &state, bool drawBox) {
    roiRect = roiRect & Rect(0, 0, frame.cols, frame.rows);
    if (roiRect.area() == 0) return;
    if (maskROI.empty()) return;

    vector<Point> rawContour = detector.findHandContour(maskROI);
    int currentFingers = 0;

    if (drawBox) {
        rectangle(frame, roiRect, Scalar(200, 200, 200), 2);
    }

    if (!rawContour.empty()) {
        vector<Point> handContour;
        for(auto pt : rawContour) handContour.push_back(pt + roiRect.tl());

        double rawRadius = 0;
        Point localCenter = detector.getPalmCenter(maskROI, rawRadius);
        Point globalCenter = localCenter + roiRect.tl();

        state.radiusBuffer.push_back(rawRadius);
        if (state.radiusBuffer.size() > state.bufferLimit) state.radiusBuffer.pop_front();

        double refRadius = rawRadius;
        if (!state.radiusBuffer.empty()) {
            refRadius = *max_element(state.radiusBuffer.begin(), state.radiusBuffer.end());
        }

        if (state.isFirstFrame) {
            state.smoothRadius = refRadius;
            state.smoothCenter = globalCenter;
            state.isFirstFrame = false;
        } else {
            double speed = (rawRadius < refRadius * 0.85) ? 0.1 : 0.4;
            state.smoothRadius = state.smoothRadius * 0.6 + refRadius * 0.4;
            state.smoothCenter.x = (int)(state.smoothCenter.x * (1 - speed) + globalCenter.x * speed);
            state.smoothCenter.y = (int)(state.smoothCenter.y * (1 - speed) + globalCenter.y * speed);
        }

        state.positionHistory.push_back(state.smoothCenter);
        if (state.positionHistory.size() > 20) state.positionHistory.pop_front();

        circle(frame, state.smoothCenter, (int)rawRadius, Scalar(0, 255, 255), 1);
        double protectionRadius = state.smoothRadius * 1.75;
        circle(frame, state.smoothCenter, (int)protectionRadius, Scalar(255, 0, 0), 2);

        vector<int> hullIndices;
        convexHull(handContour, hullIndices, false);
        vector<Point> hullPoints;
        for(int idx : hullIndices) hullPoints.push_back(handContour[idx]);

        double areaContour = contourArea(handContour);
        double areaHull = contourArea(hullPoints);
        double solidity = areaContour / areaHull;

        if (solidity > 0.94) {
            currentFingers = 0;
        }
        else {
            vector<Point> candidates;
            for (int idx : hullIndices) {
                Point pt = handContour[idx];
                if (pt.y > state.smoothCenter.y + state.smoothRadius) continue;
                if (getDist(pt, state.smoothCenter) > protectionRadius) candidates.push_back(pt);
            }

            sort(candidates.begin(), candidates.end(), [&](Point a, Point b) {
                return getDist(a, state.smoothCenter) > getDist(b, state.smoothCenter);
            });

            vector<Point> peaks;
            for (Point p : candidates) {
                bool isDuplicate = false;
                for (Point existing : peaks) {
                    if (getDist(p, existing) < state.smoothRadius * 0.8) { isDuplicate = true; break; }
                    if (getAngle(p, state.smoothCenter, existing) < 25) { isDuplicate = true; break; }
                }
                if (!isDuplicate) peaks.push_back(p);
            }

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
                if (getAngle(p_start, p_far, p_end) < 95) validDefects++;
            }

            if (validDefects >= 1) currentFingers = validDefects + 1;
            else {
                int pCount = peaks.size();
                if (pCount >= 2) {
                    currentFingers = pCount;
                    for(auto p : peaks) line(frame, state.smoothCenter, p, Scalar(0, 255, 0), 2);
                }
                else if (pCount == 1) {
                    if (solidity < 0.92) {
                        currentFingers = 1;
                        line(frame, state.smoothCenter, peaks[0], Scalar(0, 255, 0), 2);
                    } else currentFingers = 0;
                }
                else currentFingers = 0;
            }
        }

        if (currentFingers > 5) currentFingers = 5;

        vector<vector<Point>> dc = {handContour};
        drawContours(frame, dc, 0, Scalar(100, 100, 100), 1);

    } else {
        if (!state.isFirstFrame) {
             state.reset();
        }
        currentFingers = 0;
    }

    state.fingerHistory.push_back(currentFingers);
    if (state.fingerHistory.size() > 10) state.fingerHistory.pop_front();

    map<int, int> votes;
    for (int val : state.fingerHistory) votes[val]++;
    int maxVotes = 0;
    for (auto const& item : votes) {
        if (item.second > maxVotes) {
            maxVotes = item.second;
            state.displayedFingers = item.first;
        }
    }
}

struct Button {
    Rect rect;
    string text;
    Scalar color;

    Button(int x, int y, int w, int h, string t) {
        rect = Rect(x, y, w, h);
        text = t;
        color = Scalar(0, 200, 0);
    }

    void draw(Mat &frame, bool isHovered) {
        Scalar curColor = isHovered ? Scalar(0, 255, 255) : color;
        rectangle(frame, rect, curColor, 2);
        if (isHovered) rectangle(frame, rect, Scalar(0, 200, 0), -1);

        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.8;
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
    MemeDetector memeDetector;

    HandState leftHandState;
    HandState rightHandState;

    AppState appState = STATE_START_SCREEN;

    while (true) {
        Mat frame = myCam.getFrame();
        if (frame.empty()) break;

        flip(frame, frame, 1);

        Mat globalMask = detector.detectHand(frame);

        Mat bigMask;
        resize(globalMask, bigMask, Size(), 3.0, 3.0);
        imshow("HSV Camera View", bigMask);

        int w = frame.cols;
        int h = frame.rows;

        if (appState == STATE_START_SCREEN) {
            int btnY = 30;
            int btnW = 200;
            int btnH = 60;
            int gap = 20;
            int totalW = btnW * 3 + gap * 2;
            int startX = w/2 - totalW/2;

            Button mathBtn(startX, btnY, btnW, btnH, "FINGER MATH");
            Button memeBtn(startX + btnW + gap, btnY, btnW, btnH, "MEME DETECTOR");
            Button exitBtn(startX + (btnW + gap) * 2, btnY, btnW, btnH, "EXIT");

            bool pressMath = checkButtonPress(frame, mathBtn.rect, detector);
            bool pressMeme = checkButtonPress(frame, memeBtn.rect, detector);
            bool pressExit = checkButtonPress(frame, exitBtn.rect, detector);

            mathBtn.draw(frame, pressMath);
            memeBtn.draw(frame, pressMeme);
            exitBtn.draw(frame, pressExit);

            int boxSize = 450;
            if (h < 500) boxSize = h - 50;
            int padding = 20;
            Rect rectLeft(padding, h - boxSize - padding, boxSize, boxSize);
            Rect rectRight(w - boxSize - padding, h - boxSize - padding, boxSize, boxSize);

            Mat maskL = globalMask(rectLeft);
            Mat maskR = globalMask(rectRight);

            processHandInROI(frame, rectLeft, maskL, detector, leftHandState, false);
            processHandInROI(frame, rectRight, maskR, detector, rightHandState, false);

            putText(frame, "SELECT MODE", Point(w/2 - 130, h/2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 2);

            if (pressMath) {
                appState = STATE_MATH;
                leftHandState.reset();
                rightHandState.reset();
                waitKey(300);
            }
            if (pressMeme) {
                appState = STATE_MEME;
                leftHandState.reset();
                rightHandState.reset();
                waitKey(300);
            }
            if (pressExit) break;
        }

        else if (appState == STATE_MATH) {
            int boxSize = 450;
            if (h < 500) boxSize = h - 50;
            int padding = 20;
            Rect rectLeft(padding, h - boxSize - padding, boxSize, boxSize);
            Rect rectRight(w - boxSize - padding, h - boxSize - padding, boxSize, boxSize);

            Button backBtn(20, 20, 100, 40, "MENU");
            bool pressBack = checkButtonPress(frame, backBtn.rect, detector);
            backBtn.draw(frame, pressBack);

            Mat maskL = globalMask(rectLeft);
            Mat maskR = globalMask(rectRight);

            processHandInROI(frame, rectLeft, maskL, detector, leftHandState, true);
            processHandInROI(frame, rectRight, maskR, detector, rightHandState, true);

            Scalar colorL = (leftHandState.displayedFingers > 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            putText(frame, to_string(leftHandState.displayedFingers), rectLeft.tl() + Point(20, -20), FONT_HERSHEY_SIMPLEX, 2, colorL, 4);

            Scalar colorR = (rightHandState.displayedFingers > 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            putText(frame, to_string(rightHandState.displayedFingers), rectRight.tl() + Point(20, -20), FONT_HERSHEY_SIMPLEX, 2, colorR, 4);

            int totalSum = leftHandState.displayedFingers + rightHandState.displayedFingers;
            string sumText = to_string(totalSum);
            int baseLine = 0;
            Size sumSize = getTextSize(sumText, FONT_HERSHEY_SIMPLEX, 4.0, 5, &baseLine);
            int boxPadX = 40; int boxPadY = 25;
            int boxW = sumSize.width + boxPadX * 2;
            int boxH = sumSize.height + boxPadY * 2 + baseLine;
            int boxX = w / 2 - boxW / 2;
            int boxY = 80;

            rectangle(frame, Rect(boxX, boxY, boxW, boxH), Scalar(50, 50, 50), -1);
            rectangle(frame, Rect(boxX, boxY, boxW, boxH), Scalar(0, 215, 255), 4);
            putText(frame, sumText, Point(boxX + boxPadX, boxY + boxPadY + sumSize.height), FONT_HERSHEY_SIMPLEX, 4.0, Scalar(0, 255, 255), 5);

            if (pressBack) {
                appState = STATE_START_SCREEN;
                waitKey(300);
            }
        }

        else if (appState == STATE_MEME) {
            Button backBtn(20, 20, 100, 40, "MENU");
            bool pressBack = checkButtonPress(frame, backBtn.rect, detector);
            backBtn.draw(frame, pressBack);

            Rect fullLeft(20, 80, w/2 - 40, h - 100);
            Rect fullRight(w/2 + 20, 80, w/2 - 40, h - 100);

            Mat maskL = globalMask(fullLeft);
            Mat maskR = globalMask(fullRight);

            processHandInROI(frame, fullLeft, maskL, detector, leftHandState, true);
            processHandInROI(frame, fullRight, maskR, detector, rightHandState, true);

            MemeType memeL = memeDetector.detect(leftHandState.positionHistory, leftHandState.displayedFingers);
            MemeType memeR = memeDetector.detect(rightHandState.positionHistory, rightHandState.displayedFingers);

            putText(frame, "DO THE WEIGHING!", Point(w/2 - 130, 100), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200,200,200), 2);

            if (memeL == MEME_67 && memeR == MEME_67) {
                string memeText = "67";
                int face = FONT_HERSHEY_SIMPLEX;
                double scale = 15.0;
                int thick = 25;
                Size sz = getTextSize(memeText, face, scale, thick, 0);
                Point org(w/2 - sz.width/2, h/2 + sz.height/2);

                putText(frame, memeText, org, face, scale, Scalar(0, 255, 255), thick);
            }

            if (pressBack) {
                appState = STATE_START_SCREEN;
                waitKey(300);
            }
        }

        imshow("Finger Math App", frame);
        if (waitKey(1) == 27) break;
    }
    return 0;
}