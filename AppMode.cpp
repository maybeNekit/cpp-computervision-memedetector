#include "AppMode.h"
#include "MathUtils.h"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

bool checkButtonPress(Mat &frame, Rect btnRect, HandDetector &detector) {
    btnRect = btnRect & Rect(0, 0, frame.cols, frame.rows);
    if (btnRect.area() == 0) return false;
    Mat roi = frame(btnRect);
    Mat mask = detector.detectHand(roi);
    return (countNonZero(mask) > 60);
}

void processHandInROI(Mat &frame, Rect roiRect, Mat &maskROI, HandDetector &detector, HandState &state, bool drawBox) {
    roiRect = roiRect & Rect(0, 0, frame.cols, frame.rows);
    if (roiRect.area() == 0 || maskROI.empty()) return;

    vector<Point> rawContour = detector.findHandContour(maskROI);
    int currentFingers = 0;

    if (drawBox) rectangle(frame, roiRect, Scalar(200, 200, 200), 2);

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
            double speed = (rawRadius < refRadius * 0.85) ? 0.4 : 0.9;
            state.smoothRadius = state.smoothRadius * 0.3 + refRadius * 0.7;
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
        } else {
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
                } else if (pCount == 1) {
                    if (solidity < 0.92) {
                        currentFingers = 1;
                        line(frame, state.smoothCenter, peaks[0], Scalar(0, 255, 0), 2);
                    } else currentFingers = 0;
                } else currentFingers = 0;
            }
        }
        if (currentFingers > 5) currentFingers = 5;
        vector<vector<Point>> dc = {handContour};
        drawContours(frame, dc, 0, Scalar(100, 100, 100), 1);
    } else {
        if (!state.isFirstFrame) state.reset();
        currentFingers = 0;
    }
    state.updateVoting(currentFingers);
}

Button::Button(int x, int y, int w, int h, std::string t) : rect(Rect(x, y, w, h)), text(t), color(Scalar(0, 200, 0)) {}

void Button::draw(Mat &frame, bool isHovered) {
    Scalar curColor = isHovered ? Scalar(0, 255, 255) : color;
    rectangle(frame, rect, curColor, 2);
    if (isHovered) rectangle(frame, rect, Scalar(0, 200, 0), -1);
    Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.8, 2, 0);
    Point textOrg(rect.x + (rect.width - textSize.width)/2, rect.y + (rect.height + textSize.height)/2);
    putText(frame, text, textOrg, FONT_HERSHEY_SIMPLEX, 0.8, isHovered ? Scalar(255, 255, 255) : curColor, 2);
}

NextState EventVisitor::operator()(MathEvent) { waitKey(300); return NextState::GO_MATH; }
NextState EventVisitor::operator()(MemeEvent) { waitKey(300); return NextState::GO_MEME; }
NextState EventVisitor::operator()(ExitEvent) { return NextState::EXIT; }
NextState EventVisitor::operator()(NoneEvent) { return NextState::KEEP_CURRENT; }

NextState MenuMode::update(cv::Mat& frame, cv::Mat& globalMask, HandDetector& detector, int key) {
    if (key == 27) return NextState::EXIT;
    int w = frame.cols; int h = frame.rows;
    if (w == 0 || h == 0) return NextState::KEEP_CURRENT; // Защита от пустых кадров в тестах
    int btnW = 200; int btnH = 60; int gap = 20;
    int startX = w/2 - (btnW * 3 + gap * 2)/2;

    Button mathBtn(startX, 30, btnW, btnH, "FINGER MATH");
    Button memeBtn(startX + btnW + gap, 30, btnW, btnH, "MEME DETECTOR");
    Button exitBtn(startX + (btnW + gap) * 2, 30, btnW, btnH, "EXIT");

    bool pressMath = checkButtonPress(frame, mathBtn.rect, detector);
    bool pressMeme = checkButtonPress(frame, memeBtn.rect, detector);
    bool pressExit = checkButtonPress(frame, exitBtn.rect, detector);

    UIEvent event = NoneEvent{};
    if (pressMath) event = MathEvent{};
    else if (pressMeme) event = MemeEvent{};
    else if (pressExit) event = ExitEvent{};

    mathBtn.draw(frame, pressMath);
    memeBtn.draw(frame, pressMeme);
    exitBtn.draw(frame, pressExit);

    int boxSize = (h < 500) ? h - 50 : 450;
    Rect rectLeft(20, h - boxSize - 20, boxSize, boxSize);
    Rect rectRight(w - boxSize - 20, h - boxSize - 20, boxSize, boxSize);

    Mat maskL = globalMask(rectLeft);
    Mat maskR = globalMask(rectRight);

    processHandInROI(frame, rectLeft, maskL, detector, leftHandState, false);
    processHandInROI(frame, rectRight, maskR, detector, rightHandState, false);

    putText(frame, "SELECT MODE", Point(w/2 - 130, h/2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 2);

    return std::visit(EventVisitor{}, event);
}

NextState MathMode::update(cv::Mat& frame, cv::Mat& globalMask, HandDetector& detector, int key) {
    if (key == 27) return NextState::GO_MENU;
    int w = frame.cols; int h = frame.rows;
    if (w == 0 || h == 0) return NextState::KEEP_CURRENT;
    int boxSize = (h < 500) ? h - 50 : 450;
    Rect rectLeft(20, h - boxSize - 20, boxSize, boxSize);
    Rect rectRight(w - boxSize - 20, h - boxSize - 20, boxSize, boxSize);

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
    int boxX = w / 2 - (sumSize.width + 80) / 2;

    rectangle(frame, Rect(boxX, 80, sumSize.width + 80, sumSize.height + 50 + baseLine), Scalar(50, 50, 50), -1);
    rectangle(frame, Rect(boxX, 80, sumSize.width + 80, sumSize.height + 50 + baseLine), Scalar(0, 215, 255), 4);
    putText(frame, sumText, Point(boxX + 40, 80 + 25 + sumSize.height), FONT_HERSHEY_SIMPLEX, 4.0, Scalar(0, 255, 255), 5);

    if (pressBack) { waitKey(300); return NextState::GO_MENU; }
    return NextState::KEEP_CURRENT;
}

MemeMode::MemeMode(std::shared_ptr<cv::VideoCapture> v67, std::shared_ptr<cv::VideoCapture> vGriz, std::shared_ptr<cv::VideoCapture> vIvan)
    : cap67(v67), capGrizman(vGriz), capIvanzolo(vIvan) {}

void MemeMode::playVideo(std::shared_ptr<cv::VideoCapture> cap, const std::string& audioFile, int w, int h) {
    isMemePlaying = true;
    activeVideoCap = cap;
    if (activeVideoCap) activeVideoCap->set(CAP_PROP_POS_FRAMES, 0);
    namedWindow("Meme Player", WINDOW_NORMAL);
    resizeWindow("Meme Player", w, h);
    moveWindow("Meme Player", rand() % 200, rand() % 200);
    string cmd = "afplay " + audioFile + " &";
    system(cmd.c_str());
}

void MemeMode::stopVideo() {
    isMemePlaying = false;
    try { destroyWindow("Meme Player"); } catch (...) {}
    system("killall afplay 2>/dev/null");
    activeVideoCap = nullptr;
}

NextState MemeMode::update(cv::Mat& frame, cv::Mat& globalMask, HandDetector& detector, int key) {
    int w = frame.cols; int h = frame.rows;

    if (isMemePlaying && activeVideoCap != nullptr) {
        Button stopBtn(20, 20, 100, 40, "STOP");
        bool pressStop = checkButtonPress(frame, stopBtn.rect, detector);
        stopBtn.draw(frame, pressStop);

        if (key == 27 || pressStop) {
            if (pressStop) waitKey(300);
            stopVideo();
            return NextState::KEEP_CURRENT;
        }

        Mat vFrame;
        *activeVideoCap >> vFrame;
        if (vFrame.empty()) {
            stopVideo();
        } else {
            Mat displayFrame;
            resize(vFrame, displayFrame, Size(w, h));
            try {
                imshow("Meme Player", displayFrame);
                if (getWindowProperty("Meme Player", WND_PROP_VISIBLE) < 1) stopVideo();
            } catch (...) { stopVideo(); }
        }
        return NextState::KEEP_CURRENT;
    }

    if (key == 27) {
        stopVideo();
        return NextState::GO_MENU;
    }

    if (w == 0 || h == 0) return NextState::KEEP_CURRENT;

    Button backBtn(20, 20, 100, 40, "MENU");
    bool pressBack = checkButtonPress(frame, backBtn.rect, detector);
    backBtn.draw(frame, pressBack);

    Rect fullLeft(20, 80, w/2 - 40, h - 100);
    Rect fullRight(w/2 + 20, 80, w/2 - 40, h - 100);

    Mat maskL = globalMask(fullLeft);
    Mat maskR = globalMask(fullRight);

    processHandInROI(frame, fullLeft, maskL, detector, leftHandState, true);
    processHandInROI(frame, fullRight, maskR, detector, rightHandState, true);

    optional<MemeType> memeL = memeDetector.detect(leftHandState.positionHistory, leftHandState.displayedFingers);
    optional<MemeType> memeR = memeDetector.detect(rightHandState.positionHistory, rightHandState.displayedFingers);

    putText(frame, "DO A GESTURE!", Point(w/2 - 100, 100), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200,200,200), 2);

    string comboStatus = "67: " + to_string(memeTriggerCount) + "/4 | IVAN: " + to_string(ivanzoloTriggerCount) + "/3";
    putText(frame, comboStatus, Point(w/2 - 140, 140), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

    if (grizmanHoldCounter > 0) {
        int filledWidth = min(200, (int)((float)grizmanHoldCounter / GRIZMAN_HOLD_LIMIT * 200));
        rectangle(frame, Rect(w/2 - 100, 160, 200, 20), Scalar(100,100,100), 2);
        rectangle(frame, Rect(w/2 - 100, 160, filledWidth, 20), Scalar(0, 0, 255), -1);
        putText(frame, "HOLD GRIZMAN...", Point(w/2 - 80, 175), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);
    }

    int histL = leftHandState.positionHistory.size();
    int histR = rightHandState.positionHistory.size();
    bool isSyncMoveVertical = false;
    bool isSyncMoveHorizontal = false;

    if (histL > 3 && histR > 3) {
        int dyL = leftHandState.positionHistory.back().y - leftHandState.positionHistory[histL - 3].y;
        int dyR = rightHandState.positionHistory.back().y - rightHandState.positionHistory[histR - 3].y;
        int dxL = leftHandState.positionHistory.back().x - leftHandState.positionHistory[histL - 3].x;
        int dxR = rightHandState.positionHistory.back().x - rightHandState.positionHistory[histR - 3].x;

        if (dyL * dyR < 0 && abs(dyL) > 35 && abs(dyR) > 35) isSyncMoveVertical = true;
        if (dxL * dxR > 0 && abs(dxL) > 35 && abs(dxR) > 35) isSyncMoveHorizontal = true;
    }

    if (memeComboTimeout > 0) memeComboTimeout--;
    else { memeTriggerCount = 0; ivanzoloTriggerCount = 0; }

    if (moveCooldown > 0) moveCooldown--;

    if (memeL == MEME_67 && memeR == MEME_67 && isSyncMoveVertical && moveCooldown == 0) {
        memeTriggerCount++; moveCooldown = 5; memeComboTimeout = 30;
        grizmanHoldCounter = 0; ivanzoloTriggerCount = 0;
    }

    if (memeL == MEME_IVANZOLO && memeR == MEME_IVANZOLO && isSyncMoveHorizontal && moveCooldown == 0) {
        ivanzoloTriggerCount++; moveCooldown = 5; memeComboTimeout = 30;
        grizmanHoldCounter = 0; memeTriggerCount = 0;
    }

    if (memeTriggerCount >= 4) { playVideo(cap67, "67.mov", w, h); memeTriggerCount = 0; }
    if (ivanzoloTriggerCount >= 3) { playVideo(capIvanzolo, "ivanzolo.mp4", w, h); ivanzoloTriggerCount = 0; }

    if (memeL == MEME_GRIZMAN || memeR == MEME_GRIZMAN) {
        grizmanHoldCounter += 1; memeTriggerCount = 0; ivanzoloTriggerCount = 0;
    } else grizmanHoldCounter = max(0, grizmanHoldCounter - 2);

    if (grizmanHoldCounter >= GRIZMAN_HOLD_LIMIT) { playVideo(capGrizman, "grizman.mp4", w, h); grizmanHoldCounter = 0; }

    if (pressBack) { waitKey(300); stopVideo(); return NextState::GO_MENU; }
    return NextState::KEEP_CURRENT;
}