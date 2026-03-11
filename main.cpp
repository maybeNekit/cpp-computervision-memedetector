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
