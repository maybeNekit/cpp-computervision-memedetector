#pragma once
#include <opencv2/opencv.hpp>
#include <variant>
#include <memory>
#include <string>
#include "HandDetector.h"
#include "HandState.h"
#include "MemeDetector.h"

enum class NextState {
    KEEP_CURRENT,
    GO_MENU,
    GO_MATH,
    GO_MEME,
    EXIT
};

class AppMode {
public:
    virtual ~AppMode() = default;
    virtual NextState update(cv::Mat& frame, cv::Mat& globalMask, HandDetector& detector, int key) = 0;
};

// --- Вспомогательные функции и структуры ---
bool checkButtonPress(cv::Mat &frame, cv::Rect btnRect, HandDetector &detector);
void processHandInROI(cv::Mat &frame, cv::Rect roiRect, cv::Mat &maskROI, HandDetector &detector, HandState &state, bool drawBox);

struct Button {
    cv::Rect rect;
    std::string text;
    cv::Scalar color;

    Button(int x, int y, int w, int h, std::string t);
    void draw(cv::Mat &frame, bool isHovered);
};

struct MathEvent {};
struct MemeEvent {};
struct ExitEvent {};
struct NoneEvent {};
using UIEvent = std::variant<MathEvent, MemeEvent, ExitEvent, NoneEvent>;

struct EventVisitor {
    NextState operator()(MathEvent);
    NextState operator()(MemeEvent);
    NextState operator()(ExitEvent);
    NextState operator()(NoneEvent);
};

// --- Режимы приложения ---
class MenuMode : public AppMode {
private:
    HandState leftHandState;
    HandState rightHandState;
public:
    NextState update(cv::Mat& frame, cv::Mat& globalMask, HandDetector& detector, int key) override;
};

class MathMode : public AppMode {
private:
    HandState leftHandState;
    HandState rightHandState;
public:
    NextState update(cv::Mat& frame, cv::Mat& globalMask, HandDetector& detector, int key) override;
};

class MemeMode : public AppMode {
private:
    HandState leftHandState;
    HandState rightHandState;
    MemeDetector memeDetector;
    std::shared_ptr<cv::VideoCapture> cap67;
    std::shared_ptr<cv::VideoCapture> capGrizman;
    std::shared_ptr<cv::VideoCapture> capIvanzolo;
    std::shared_ptr<cv::VideoCapture> activeVideoCap = nullptr;

    bool isMemePlaying = false;
    int memeTriggerCount = 0;
    int ivanzoloTriggerCount = 0;
    int moveCooldown = 0;
    int memeComboTimeout = 0;
    int grizmanHoldCounter = 0;
    const int GRIZMAN_HOLD_LIMIT = 20;

    void playVideo(std::shared_ptr<cv::VideoCapture> cap, const std::string& audioFile, int w, int h);
    void stopVideo();

public:
    MemeMode(std::shared_ptr<cv::VideoCapture> v67, std::shared_ptr<cv::VideoCapture> vGriz, std::shared_ptr<cv::VideoCapture> vIvan);
    NextState update(cv::Mat& frame, cv::Mat& globalMask, HandDetector& detector, int key) override;
};