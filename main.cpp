#include <iostream>
#include <cmath>
#include <numeric>
#include <deque>
#include <vector>
#include <algorithm>
#include <map>
#include <cstdlib> 
#include <ctime>
#include <memory> 
#include <optional>
#include <variant>

#include "Camera.h"
#include "HandDetector.h"
#include "MemeDetector.h"
#include "AppMode.h"

using namespace cv;
using namespace std;

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

constexpr double pi() { return 3.14159265358979323846; }

double getAngle(Point s, Point f, Point e) {
    double l1 = norm(f - s);
    double l2 = norm(f - e);
    double dot = (s.x - f.x) * (e.x - f.x) + (s.y - f.y) * (e.y - f.y);
    double angle = acos(dot / (l1 * l2));
    return angle * 180.0 / pi();
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
    Rect rect; string text; Scalar color;
    Button(int x, int y, int w, int h, string t) : rect(Rect(x, y, w, h)), text(t), color(Scalar(0, 200, 0)) {}

    void draw(Mat &frame, bool isHovered) {
        Scalar curColor = isHovered ? Scalar(0, 255, 255) : color;
        rectangle(frame, rect, curColor, 2);
        if (isHovered) rectangle(frame, rect, Scalar(0, 200, 0), -1);

        Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.8, 2, 0);
        Point textOrg(rect.x + (rect.width - textSize.width)/2, rect.y + (rect.height + textSize.height)/2);
        putText(frame, text, textOrg, FONT_HERSHEY_SIMPLEX, 0.8, isHovered ? Scalar(255, 255, 255) : curColor, 2);
    }
};

struct MathEvent {};
struct MemeEvent {};
struct ExitEvent {};
struct NoneEvent {};
using UIEvent = std::variant<MathEvent, MemeEvent, ExitEvent, NoneEvent>;

struct EventVisitor {
    NextState operator()(MathEvent) { waitKey(300); return NextState::GO_MATH; }
    NextState operator()(MemeEvent) { waitKey(300); return NextState::GO_MEME; }
    NextState operator()(ExitEvent) { return NextState::EXIT; }
    NextState operator()(NoneEvent) { return NextState::KEEP_CURRENT; }
};

class MenuMode : public AppMode {
private:
    HandState leftHandState;
    HandState rightHandState;
public:
    NextState update(cv::Mat& frame, cv::Mat& globalMask, HandDetector& detector) override {
        int w = frame.cols; int h = frame.rows;
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
};

class MathMode : public AppMode {
private:
    HandState leftHandState;
    HandState rightHandState;
public:
    NextState update(cv::Mat& frame, cv::Mat& globalMask, HandDetector& detector) override {
        int w = frame.cols; int h = frame.rows;
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
};

class MemeMode : public AppMode {
private:
    HandState leftHandState;
    HandState rightHandState;
    MemeDetector memeDetector;
    shared_ptr<VideoCapture> cap67;
    shared_ptr<VideoCapture> capGrizman;
    shared_ptr<VideoCapture> activeVideoCap = nullptr;

    bool isMemePlaying = false;
    int memeTriggerCount = 0;
    int moveCooldown = 0;
    int memeComboTimeout = 0;
    int grizmanHoldCounter = 0;
    const int GRIZMAN_HOLD_LIMIT = 20;

public:
    MemeMode(shared_ptr<VideoCapture> v67, shared_ptr<VideoCapture> vGriz) : cap67(v67), capGrizman(vGriz) {}

    NextState update(cv::Mat& frame, cv::Mat& globalMask, HandDetector& detector) override {
        int w = frame.cols; int h = frame.rows;

        if (isMemePlaying && activeVideoCap != nullptr) {
            Mat vFrame;
            *activeVideoCap >> vFrame;
            if (vFrame.empty()) {
                stopVideo();
            } else {
                Mat displayFrame;
                resize(vFrame, displayFrame, Size(w, h));
                imshow("Meme Player", displayFrame);
                if (getWindowProperty("Meme Player", WND_PROP_VISIBLE) < 1) stopVideo();
            }
            return NextState::KEEP_CURRENT;
        }

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
        putText(frame, "WEIGH: " + to_string(memeTriggerCount) + "/4", Point(w/2 - 80, 140), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

        if (grizmanHoldCounter > 0) {
            int filledWidth = min(200, (int)((float)grizmanHoldCounter / GRIZMAN_HOLD_LIMIT * 200));
            rectangle(frame, Rect(w/2 - 100, 160, 200, 20), Scalar(100,100,100), 2);
            rectangle(frame, Rect(w/2 - 100, 160, filledWidth, 20), Scalar(0, 0, 255), -1);
            putText(frame, "HOLD GRIZMAN...", Point(w/2 - 80, 175), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);
        }

        int histL = leftHandState.positionHistory.size();
        int histR = rightHandState.positionHistory.size();
        bool isSyncMove = false;
        if (histL > 3 && histR > 3) {
            int dyL = leftHandState.positionHistory.back().y - leftHandState.positionHistory[histL - 3].y;
            int dyR = rightHandState.positionHistory.back().y - rightHandState.positionHistory[histR - 3].y;
            if (dyL * dyR < 0 && abs(dyL) > 35 && abs(dyR) > 35) isSyncMove = true;
        }

        if (memeComboTimeout > 0) memeComboTimeout--; else memeTriggerCount = 0;
        if (moveCooldown > 0) moveCooldown--;

        if (memeL == MEME_67 && memeR == MEME_67 && isSyncMove && moveCooldown == 0) {
            memeTriggerCount++;
            moveCooldown = 5;
            memeComboTimeout = 30;
            grizmanHoldCounter = 0;
        }

        if (memeTriggerCount >= 4) {
            playVideo(cap67, "67.mov", w, h);
            memeTriggerCount = 0;
        }

        if (memeL == MEME_GRIZMAN || memeR == MEME_GRIZMAN) {
            grizmanHoldCounter += 1;
            memeTriggerCount = 0;
        } else {
            grizmanHoldCounter = max(0, grizmanHoldCounter - 2);
        }

        if (grizmanHoldCounter >= GRIZMAN_HOLD_LIMIT) {
            playVideo(capGrizman, "grizman.mp4", w, h);
            grizmanHoldCounter = 0;
        }

        if (pressBack) { waitKey(300); stopVideo(); return NextState::GO_MENU; }
        return NextState::KEEP_CURRENT;
    }

private:
    void playVideo(shared_ptr<VideoCapture> cap, const string& audioFile, int w, int h) {
        isMemePlaying = true;
        activeVideoCap = cap;
        activeVideoCap->set(CAP_PROP_POS_FRAMES, 0);
        namedWindow("Meme Player", WINDOW_NORMAL);
        resizeWindow("Meme Player", w, h);
        moveWindow("Meme Player", rand() % 200, rand() % 200);
        string cmd = "afplay " + audioFile + " &";
        system(cmd.c_str());
    }

    void stopVideo() {
        isMemePlaying = false;
        if (getWindowProperty("Meme Player", WND_PROP_VISIBLE) >= 0) destroyWindow("Meme Player");
        system("killall afplay 2>/dev/null");
        activeVideoCap = nullptr;
    }
};

int main() {
    srand(time(0));
    try {
        Camera myCam;
        HandDetector detector;

        auto cap67 = make_shared<VideoCapture>("67.mov");
        auto capGrizman = make_shared<VideoCapture>("grizman.mp4");

        unique_ptr<AppMode> currentMode = make_unique<MenuMode>();

        namedWindow("Finger Math App", WINDOW_NORMAL);

        while (true) {
            Mat frame = myCam.getFrame();
            if (frame.empty()) break;
            flip(frame, frame, 1);

            Mat globalMask = detector.detectHand(frame);

            NextState next = currentMode->update(frame, globalMask, detector);

            if (next == NextState::GO_MATH) {
                currentMode = make_unique<MathMode>();
            } else if (next == NextState::GO_MEME) {
                currentMode = make_unique<MemeMode>(cap67, capGrizman);
            } else if (next == NextState::GO_MENU) {
                currentMode = make_unique<MenuMode>();
            } else if (next == NextState::EXIT) {
                break;
            }

            imshow("Finger Math App", frame);
            if (waitKey(1) == 27) break;
        }

        system("killall afplay 2>/dev/null");
    } catch (const CameraException& e) {
        cerr << "Camera Init Error: " << e.what() << endl;
        return -1;
    } catch (const exception& e) {
        cerr << "Fatal Error: " << e.what() << endl;
        return -1;
    }
    return 0;
}