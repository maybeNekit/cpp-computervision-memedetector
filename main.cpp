#include <iostream>
#include <cmath>
#include <numeric>
#include <deque>
#include <vector>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <ctime>
#include "Camera.h"
#include "HandDetector.h"
#include "MemeDetector.h"

enum AppState {
    STATE_START_SCREEN,
    STATE_MATH,
    STATE_MEME
};

struct HandState {
    std::deque<int> fingerHistory;
    std::deque<double> radiusBuffer;
    std::deque<cv::Point> positionHistory;
    static const int bufferLimit = 100;

    cv::Point smoothCenter = cv::Point(0, 0);
    double smoothRadius = 0;
    bool isFirstFrame = true;

    int displayedFingers = 0;

    void reset() {
        fingerHistory.clear();
        radiusBuffer.clear();
        positionHistory.clear();
        smoothCenter = cv::Point(0, 0);
        smoothRadius = 0;
        isFirstFrame = true;
        displayedFingers = 0;
    }
};

double getAngle(cv::Point s, cv::Point f, cv::Point e) {
    double l1 = cv::norm(f - s);
    double l2 = cv::norm(f - e);
    double dot = (s.x - f.x) * (e.x - f.x) + (s.y - f.y) * (e.y - f.y);
    double angle = std::acos(dot / (l1 * l2));
    return angle * 180.0 / 3.14159265;
}

double getDist(cv::Point p1, cv::Point p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

bool checkButtonPress(cv::Mat &frame, cv::Rect btnRect, HandDetector &detector) {
    btnRect = btnRect & cv::Rect(0, 0, frame.cols, frame.rows);
    if (btnRect.area() == 0) return false;
    cv::Mat roi = frame(btnRect);
    cv::Mat mask = detector.detectHand(roi);
    int skinPixels = cv::countNonZero(mask);
    return (skinPixels > 60);
}

void processHandInROI(cv::Mat &frame, cv::Rect roiRect, cv::Mat &maskROI, HandDetector &detector, HandState &state, bool drawBox) {
    roiRect = roiRect & cv::Rect(0, 0, frame.cols, frame.rows);
    if (roiRect.area() == 0) return;
    if (maskROI.empty()) return;

    std::vector<cv::Point> rawContour = detector.findHandContour(maskROI);
    int currentFingers = 0;

    if (drawBox) {
        cv::rectangle(frame, roiRect, cv::Scalar(200, 200, 200), 2);
    }

    if (!rawContour.empty()) {
        std::vector<cv::Point> handContour;
        for(auto pt : rawContour) handContour.push_back(pt + roiRect.tl());

        double rawRadius = 0;
        cv::Point localCenter = detector.getPalmCenter(maskROI, rawRadius);
        cv::Point globalCenter = localCenter + roiRect.tl();

        state.radiusBuffer.push_back(rawRadius);
        if (state.radiusBuffer.size() > state.bufferLimit) state.radiusBuffer.pop_front();

        double refRadius = rawRadius;
        if (!state.radiusBuffer.empty()) {
            refRadius = *std::max_element(state.radiusBuffer.begin(), state.radiusBuffer.end());
        }

        if (state.isFirstFrame) {
            state.smoothRadius = refRadius;
            state.smoothCenter = globalCenter;
            state.isFirstFrame = false;
        } else {
            double speed = (rawRadius < refRadius * 0.85) ? 0.4 : 0.9;
            state.smoothRadius = state.smoothRadius * 0.3 + refRadius * 0.7;
            state.smoothCenter.x = (int)(state.smoothCenter.x * (1 - speed) + globalCenter.x * speed);
            state.smoothCenter.y = (int)(state.smoothCenter.y * (1 - speed) + globalCenter.y * speed);
        }

        state.positionHistory.push_back(state.smoothCenter);
        if (state.positionHistory.size() > 20) state.positionHistory.pop_front();

        cv::circle(frame, state.smoothCenter, (int)rawRadius, cv::Scalar(0, 255, 255), 1);
        double protectionRadius = state.smoothRadius * 1.75;
        cv::circle(frame, state.smoothCenter, (int)protectionRadius, cv::Scalar(255, 0, 0), 2);

        std::vector<int> hullIndices;
        cv::convexHull(handContour, hullIndices, false);
        std::vector<cv::Point> hullPoints;
        for(int idx : hullIndices) hullPoints.push_back(handContour[idx]);

        double areaContour = cv::contourArea(handContour);
        double areaHull = cv::contourArea(hullPoints);
        double solidity = areaContour / areaHull;

        if (solidity > 0.94) {
            currentFingers = 0;
        }
        else {
            std::vector<cv::Point> candidates;
            for (int idx : hullIndices) {
                cv::Point pt = handContour[idx];
                if (pt.y > state.smoothCenter.y + state.smoothRadius) continue;
                if (getDist(pt, state.smoothCenter) > protectionRadius) candidates.push_back(pt);
            }

            std::sort(candidates.begin(), candidates.end(), [&](cv::Point a, cv::Point b) {
                return getDist(a, state.smoothCenter) > getDist(b, state.smoothCenter);
            });

            std::vector<cv::Point> peaks;
            for (cv::Point p : candidates) {
                bool isDuplicate = false;
                for (cv::Point existing : peaks) {
                    if (getDist(p, existing) < state.smoothRadius * 0.8) { isDuplicate = true; break; }
                    if (getAngle(p, state.smoothCenter, existing) < 25) { isDuplicate = true; break; }
                }
                if (!isDuplicate) peaks.push_back(p);
            }

            std::vector<cv::Vec4i> defects;
            if (hullIndices.size() > 3) {
                try { cv::convexityDefects(handContour, hullIndices, defects); } catch (...) {}
            }
            int validDefects = 0;
            for (int i = 0; i < defects.size(); i++) {
                cv::Point p_start = handContour[defects[i][0]];
                cv::Point p_end = handContour[defects[i][1]];
                cv::Point p_far = handContour[defects[i][2]];
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
                    for(auto p : peaks) cv::line(frame, state.smoothCenter, p, cv::Scalar(0, 255, 0), 2);
                }
                else if (pCount == 1) {
                    if (solidity < 0.92) {
                        currentFingers = 1;
                        cv::line(frame, state.smoothCenter, peaks[0], cv::Scalar(0, 255, 0), 2);
                    } else currentFingers = 0;
                }
                else currentFingers = 0;
            }
        }

        if (currentFingers > 5) currentFingers = 5;

        std::vector<std::vector<cv::Point>> dc = {handContour};
        cv::drawContours(frame, dc, 0, cv::Scalar(100, 100, 100), 1);

    } else {
        if (!state.isFirstFrame) {
             state.reset();
        }
        currentFingers = 0;
    }

    state.fingerHistory.push_back(currentFingers);
    if (state.fingerHistory.size() > 10) state.fingerHistory.pop_front();

    std::map<int, int> votes;
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
    cv::Rect rect;
    std::string text;
    cv::Scalar color;

    Button(int x, int y, int w, int h, std::string t) {
        rect = cv::Rect(x, y, w, h);
        text = t;
        color = cv::Scalar(0, 200, 0);
    }

    void draw(cv::Mat &frame, bool isHovered) {
        cv::Scalar curColor = isHovered ? cv::Scalar(0, 255, 255) : color;
        cv::rectangle(frame, rect, curColor, 2);
        if (isHovered) cv::rectangle(frame, rect, cv::Scalar(0, 200, 0), -1);

        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.8;
        int thickness = 2;
        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, 0);
        cv::Point textOrg(rect.x + (rect.width - textSize.width)/2, rect.y + (rect.height + textSize.height)/2);

        cv::Scalar textColor = isHovered ? cv::Scalar(255, 255, 255) : curColor;
        cv::putText(frame, text, textOrg, fontFace, fontScale, textColor, thickness);
    }
};

int main() {
    std::srand(std::time(0));
    Camera myCam;
    HandDetector detector;
    MemeDetector memeDetector;

    cv::VideoCapture cap67("67.mov");

    HandState leftHandState;
    HandState rightHandState;

    AppState appState = STATE_START_SCREEN;
    bool isMemePlaying = false;
    bool windowSetup = false;

    int memeTriggerCount = 0;
    int moveCooldown = 0;
    int memeComboTimeout = 0;

    while (true) {
        cv::Mat frame = myCam.getFrame();
        if (frame.empty()) break;

        cv::flip(frame, frame, 1);
        int w = frame.cols;
        int h = frame.rows;

        if (!windowSetup) {
            cv::namedWindow("Finger Math App", cv::WINDOW_NORMAL);
            cv::resizeWindow("Finger Math App", w, h);
            cv::moveWindow("Finger Math App", 0, 0);
            windowSetup = true;
        }

        if (isMemePlaying) {
            cv::Mat vFrame;
            cap67 >> vFrame;

            if (vFrame.empty()) {
                isMemePlaying = false;
                cv::destroyWindow("Meme Player");
                std::system("killall afplay");
            } else {
                cv::Mat displayFrame;
                cv::resize(vFrame, displayFrame, cv::Size(w, h));

                cv::imshow("Meme Player", displayFrame);

                if (cv::getWindowProperty("Meme Player", cv::WND_PROP_VISIBLE) < 1) {
                     isMemePlaying = false;
                     cv::destroyWindow("Meme Player");
                     std::system("killall afplay");
                }
            }

            cv::imshow("Finger Math App", frame);

            int key = cv::waitKey(30);
            if (key == 27) {
                isMemePlaying = false;
                cv::destroyWindow("Meme Player");
                std::system("killall afplay");
            }
            continue;
        }

        cv::Mat globalMask = detector.detectHand(frame);

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
            cv::Rect rectLeft(padding, h - boxSize - padding, boxSize, boxSize);
            cv::Rect rectRight(w - boxSize - padding, h - boxSize - padding, boxSize, boxSize);

            cv::Mat maskL = globalMask(rectLeft);
            cv::Mat maskR = globalMask(rectRight);

            processHandInROI(frame, rectLeft, maskL, detector, leftHandState, false);
            processHandInROI(frame, rectRight, maskR, detector, rightHandState, false);

            cv::putText(frame, "SELECT MODE", cv::Point(w/2 - 130, h/2), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255), 2);

            if (pressMath) {
                appState = STATE_MATH;
                leftHandState.reset();
                rightHandState.reset();
                cv::waitKey(300);
            }
            if (pressMeme) {
                appState = STATE_MEME;
                leftHandState.reset();
                rightHandState.reset();
                isMemePlaying = false;
                memeTriggerCount = 0;
                moveCooldown = 0;
                memeComboTimeout = 0;
                cv::waitKey(300);
            }
            if (pressExit) break;
        }

        else if (appState == STATE_MATH) {
            int boxSize = 450;
            if (h < 500) boxSize = h - 50;
            int padding = 20;
            cv::Rect rectLeft(padding, h - boxSize - padding, boxSize, boxSize);
            cv::Rect rectRight(w - boxSize - padding, h - boxSize - padding, boxSize, boxSize);

            Button backBtn(20, 20, 100, 40, "MENU");
            bool pressBack = checkButtonPress(frame, backBtn.rect, detector);
            backBtn.draw(frame, pressBack);

            cv::Mat maskL = globalMask(rectLeft);
            cv::Mat maskR = globalMask(rectRight);

            processHandInROI(frame, rectLeft, maskL, detector, leftHandState, true);
            processHandInROI(frame, rectRight, maskR, detector, rightHandState, true);

            cv::Scalar colorL = (leftHandState.displayedFingers > 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::putText(frame, std::to_string(leftHandState.displayedFingers), rectLeft.tl() + cv::Point(20, -20), cv::FONT_HERSHEY_SIMPLEX, 2, colorL, 4);

            cv::Scalar colorR = (rightHandState.displayedFingers > 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::putText(frame, std::to_string(rightHandState.displayedFingers), rectRight.tl() + cv::Point(20, -20), cv::FONT_HERSHEY_SIMPLEX, 2, colorR, 4);

            int totalSum = leftHandState.displayedFingers + rightHandState.displayedFingers;
            std::string sumText = std::to_string(totalSum);
            int baseLine = 0;
            cv::Size sumSize = cv::getTextSize(sumText, cv::FONT_HERSHEY_SIMPLEX, 4.0, 5, &baseLine);
            int boxPadX = 40; int boxPadY = 25;
            int boxW = sumSize.width + boxPadX * 2;
            int boxH = sumSize.height + boxPadY * 2 + baseLine;
            int boxX = w / 2 - boxW / 2;
            int boxY = 80;

            cv::rectangle(frame, cv::Rect(boxX, boxY, boxW, boxH), cv::Scalar(50, 50, 50), -1);
            cv::rectangle(frame, cv::Rect(boxX, boxY, boxW, boxH), cv::Scalar(0, 215, 255), 4);
            cv::putText(frame, sumText, cv::Point(boxX + boxPadX, boxY + boxPadY + sumSize.height), cv::FONT_HERSHEY_SIMPLEX, 4.0, cv::Scalar(0, 255, 255), 5);

            if (pressBack) {
                appState = STATE_START_SCREEN;
                cv::waitKey(300);
            }
        }

        else if (appState == STATE_MEME) {
            Button backBtn(20, 20, 100, 40, "MENU");
            bool pressBack = checkButtonPress(frame, backBtn.rect, detector);
            backBtn.draw(frame, pressBack);

            cv::Rect fullLeft(20, 80, w/2 - 40, h - 100);
            cv::Rect fullRight(w/2 + 20, 80, w/2 - 40, h - 100);

            cv::Mat maskL = globalMask(fullLeft);
            cv::Mat maskR = globalMask(fullRight);

            processHandInROI(frame, fullLeft, maskL, detector, leftHandState, true);
            processHandInROI(frame, fullRight, maskR, detector, rightHandState, true);

            MemeType memeL = memeDetector.detect(leftHandState.positionHistory, leftHandState.displayedFingers);
            MemeType memeR = memeDetector.detect(rightHandState.positionHistory, rightHandState.displayedFingers);

            cv::putText(frame, "DO THE WEIGHING!", cv::Point(w/2 - 130, 100), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200,200,200), 2);

            std::string comboText = "COMBO: " + std::to_string(memeTriggerCount) + "/3";
            cv::putText(frame, comboText, cv::Point(w/2 - 80, 150), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);

            int histL = leftHandState.positionHistory.size();
            int histR = rightHandState.positionHistory.size();

            bool isSyncMove = false;

            if (histL > 3 && histR > 3) {
                int dyL = leftHandState.positionHistory.back().y - leftHandState.positionHistory[histL - 3].y;
                int dyR = rightHandState.positionHistory.back().y - rightHandState.positionHistory[histR - 3].y;

                if (dyL * dyR < 0 && std::abs(dyL) > 35 && std::abs(dyR) > 35) {
                    isSyncMove = true;
                }
            }

            if (memeComboTimeout > 0) memeComboTimeout--;
            else memeTriggerCount = 0;

            if (moveCooldown > 0) moveCooldown--;

            if (memeL == MEME_67 && memeR == MEME_67 && isSyncMove) {
                if (moveCooldown == 0) {
                    memeTriggerCount++;
                    moveCooldown = 5;
                    memeComboTimeout = 30;
                }
            }

            if (memeTriggerCount >= 3) {
                isMemePlaying = true;
                cap67.set(cv::CAP_PROP_POS_FRAMES, 0);

                cv::namedWindow("Meme Player", cv::WINDOW_NORMAL);
                cv::resizeWindow("Meme Player", w, h);
                int rx = std::rand() % 200;
                int ry = std::rand() % 200;
                cv::moveWindow("Meme Player", rx, ry);

                std::system("afplay 67.mov &");
                memeTriggerCount = 0;
            }

            if (pressBack) {
                appState = STATE_START_SCREEN;
                cv::waitKey(300);
            }
        }

        cv::imshow("Finger Math App", frame);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}