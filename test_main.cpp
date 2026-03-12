#include <gtest/gtest.h>
#include "MemeDetector.h"
#include "MathUtils.h"
#include "HandState.h"
#include "AppMode.h"
#include "HandDetector.h"
#include "Camera.h"

using namespace cv;
using namespace std;

// --- ТЕСТЫ МАТЕМАТИКИ ---
TEST(MathUtilsTest, CalculateDistance) {
    EXPECT_DOUBLE_EQ(getDist(Point(0,0), Point(3,4)), 5.0);
    EXPECT_DOUBLE_EQ(getDist(Point(0,0), Point(0,10)), 10.0);
}

TEST(MathUtilsTest, CalculateAngle) {
    Point s(0, 10); Point f(0, 0); Point e(10, 0);
    EXPECT_NEAR(getAngle(s, f, e), 90.0, 1e-5);
    EXPECT_DOUBLE_EQ(getAngle(Point(0,0), Point(0,0), Point(0,0)), 0.0);
}

// --- ТЕСТЫ СОСТОЯНИЯ РУКИ ---
TEST(HandStateTest, ResetClearsData) {
    HandState state;
    state.fingerHistory.push_back(5);
    state.reset();
    EXPECT_TRUE(state.fingerHistory.empty());
}

TEST(HandStateTest, VotingLogicFiltersNoise) {
    HandState state;
    state.updateVoting(2); state.updateVoting(2); state.updateVoting(5); state.updateVoting(2);
    EXPECT_EQ(state.displayedFingers, 2);
}

// --- ТЕСТЫ МЕМОВ ---
TEST(MemeDetectorTest, DetectMeme67_VerticalMotion) {
    MemeDetector detector;
    deque<Point> history = {Point(100, 100), Point(100, 80), Point(100, 60), Point(100, 40), Point(100, 20)};
    auto result = detector.detect(history, 5);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), MEME_67);
}

TEST(MemeDetectorTest, DetectIvanZolo_HorizontalMotion) {
    MemeDetector detector;
    deque<Point> history = {Point(20, 100), Point(40, 100), Point(60, 100), Point(80, 100), Point(100, 100)};
    auto result = detector.detect(history, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), MEME_IVANZOLO);
}

TEST(MemeDetectorTest, DetectGrizman_StaticMotion) {
    MemeDetector detector;
    deque<Point> history = {Point(100, 100), Point(105, 102), Point(98, 99), Point(102, 105), Point(100, 100)};
    auto result = detector.detect(history, 2);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), MEME_GRIZMAN);
}

// --- НОВЫЕ ТЕСТЫ: РАБОТА С ИЗОБРАЖЕНИЯМИ (HandDetector) ---
TEST(HandDetectorTest, EmptyImageReturnsEmptyMask) {
    HandDetector detector;
    Mat emptyFrame = Mat::zeros(100, 100, CV_8UC3);
    Mat mask = detector.detectHand(emptyFrame);
    EXPECT_EQ(countNonZero(mask), 0);
}

TEST(HandDetectorTest, EmptyMaskReturnsZeroCenter) {
    HandDetector detector;
    Mat emptyMask = Mat::zeros(100, 100, CV_8UC1);
    double radius = 0;
    Point center = detector.getPalmCenter(emptyMask, radius);
    EXPECT_EQ(center.x, 0);
    EXPECT_EQ(center.y, 0);
    EXPECT_EQ(radius, 0.0);
}

TEST(HandDetectorTest, EmptyMaskReturnsEmptyContour) {
    HandDetector detector;
    Mat emptyMask = Mat::zeros(100, 100, CV_8UC1);
    auto contour = detector.findHandContour(emptyMask);
    EXPECT_TRUE(contour.empty());
}

// --- НОВЫЕ ТЕСТЫ: ПЕРЕХОДЫ МЕЖДУ ЭКРАНАМИ (AppModes) ---
TEST(MenuModeTest, EscKeyReturnsExitState) {
    MenuMode menu;
    HandDetector detector;
    Mat frame = Mat::zeros(480, 640, CV_8UC3);
    Mat mask = Mat::zeros(480, 640, CV_8UC1);

    // Передаем нажатие клавиши ESC (код 27)
    NextState state = menu.update(frame, mask, detector, 27);
    EXPECT_EQ(state, NextState::EXIT);
}

TEST(MathModeTest, EscKeyReturnsToMenu) {
    MathMode math;
    HandDetector detector;
    Mat frame = Mat::zeros(480, 640, CV_8UC3);
    Mat mask = Mat::zeros(480, 640, CV_8UC1);

    NextState state = math.update(frame, mask, detector, 27);
    EXPECT_EQ(state, NextState::GO_MENU);
}

TEST(MemeModeTest, EscKeyReturnsToMenu) {
    // Имитируем пустые указатели на видео (чтобы не открывались реальные файлы)
    auto v1 = make_shared<VideoCapture>();
    auto v2 = make_shared<VideoCapture>();
    auto v3 = make_shared<VideoCapture>();

    MemeMode meme(v1, v2, v3);
    HandDetector detector;
    Mat frame = Mat::zeros(480, 640, CV_8UC3);
    Mat mask = Mat::zeros(480, 640, CV_8UC1);

    NextState state = meme.update(frame, mask, detector, 27);
    EXPECT_EQ(state, NextState::GO_MENU);
}

TEST(EventVisitorTest, VisitsCorrectly) {
    EventVisitor visitor;
    // NoneEvent означает, что ни одна кнопка не нажата -> нужно оставаться на текущем экране
    EXPECT_EQ(visitor(NoneEvent{}), NextState::KEEP_CURRENT);
    EXPECT_EQ(visitor(ExitEvent{}), NextState::EXIT);
}

// --- ТЕСТЫ ОБОРУДОВАНИЯ (Camera) ---
TEST(CameraTest, CameraInitializationAndCapture) {
    try {
        Camera cam;
        Mat frame = cam.getFrame();
        // Если камера есть, метод getFrame() не должен ронять программу.
        // Кадр может быть пустым (если камера только запускается), но краша быть не должно.
        SUCCEED() << "Камера успешно инициализирована и отдала кадр.";
    } catch (const CameraException& e) {
        // Если камеры нет (или тест запущен на сервере без вебки),
        // ожидается выброс исключения. Это тоже правильное поведение.
        SUCCEED() << "Камера не найдена, выброшено ожидаемое исключение: " << e.what();
    }
}

// --- ТЕСТЫ ОТРИСОВКИ ИНТЕРФЕЙСА (UI / Ocv Draw) ---
TEST(UITest, ButtonDrawModifiesPixels) {
    // Создаем абсолютно черный кадр
    Mat frame = Mat::zeros(200, 200, CV_8UC3);
    Button btn(10, 10, 100, 50, "TEST");

    // Вызываем отрисовку кнопки (без наведения)
    btn.draw(frame, false);

    // Преобразуем кадр в 1-канальный, чтобы посчитать ненулевые пиксели
    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

    // Если отрисовка сработала, на кадре должны появиться цветные пиксели (рамка и текст)
    EXPECT_GT(countNonZero(grayFrame), 0);
}

TEST(UITest, ButtonDrawHoverModifiesPixels) {
    Mat frame = Mat::zeros(200, 200, CV_8UC3);
    Button btn(10, 10, 100, 50, "TEST");

    // Вызываем отрисовку кнопки (С наведением)
    btn.draw(frame, true);

    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

    // При наведении кнопка заливается цветом, пикселей должно быть изменено больше нуля
    EXPECT_GT(countNonZero(grayFrame), 0);
}

TEST(UITest, ProcessHandInROIDrawsBox) {
    // Проверяем, что функция processHandInROI корректно рисует серую рамку (drawBox = true)
    Mat frame = Mat::zeros(300, 300, CV_8UC3);
    Mat mask = Mat::zeros(300, 300, CV_8UC1); // Пустая маска
    HandDetector detector;
    HandState state;
    Rect roi(50, 50, 100, 100);

    processHandInROI(frame, roi, mask, detector, state, true);

    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

    // Даже если рука не найдена, рамка ROI должна быть нарисована
    EXPECT_GT(countNonZero(grayFrame), 0);
}