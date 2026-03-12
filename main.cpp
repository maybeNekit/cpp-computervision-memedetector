#include <iostream>
#include <memory>
#include "Camera.h"
#include "HandDetector.h"
#include "AppMode.h"

using namespace cv;
using namespace std;

int main() {
    srand(time(0));
    try {
        Camera myCam;
        HandDetector detector;

        auto cap67 = make_shared<VideoCapture>("67.mov");
        auto capGrizman = make_shared<VideoCapture>("grizman.mp4");
        auto capIvanzolo = make_shared<VideoCapture>("ivanzolo.mp4");

        unique_ptr<AppMode> currentMode = make_unique<MenuMode>();
        namedWindow("Finger Math App", WINDOW_NORMAL);
        int key = -1;

        while (true) {
            Mat frame = myCam.getFrame();
            if (frame.empty()) break;
            flip(frame, frame, 1);

            Mat globalMask = detector.detectHand(frame);
            NextState next = currentMode->update(frame, globalMask, detector, key);

            if (next == NextState::GO_MATH) currentMode = make_unique<MathMode>();
            else if (next == NextState::GO_MEME) currentMode = make_unique<MemeMode>(cap67, capGrizman, capIvanzolo);
            else if (next == NextState::GO_MENU) currentMode = make_unique<MenuMode>();
            else if (next == NextState::EXIT) break;

            imshow("Finger Math App", frame);
            key = waitKey(1);
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