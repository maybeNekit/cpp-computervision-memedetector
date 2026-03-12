#include <gtest/gtest.h>
#include "MemeDetector.h"
#include <deque>

using namespace cv;
using namespace std;


TEST(MemeDetectorTest, InsufficientHistory) {
    MemeDetector detector;
    deque<Point> history = {Point(0,0), Point(1,1), Point(2,2)};

    auto result = detector.detect(history, 0);
    EXPECT_FALSE(result.has_value());
}

// 2. 67
TEST(MemeDetectorTest, DetectMeme67_VerticalMotion) {
    MemeDetector detector;

    deque<Point> history = {
        Point(100, 100),
        Point(100, 80),
        Point(100, 60),
        Point(100, 40),
        Point(100, 20)
    };

    // 5 пальцев
    auto result = detector.detect(history, 5);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), MEME_67);
}

// 3. Иван золо
TEST(MemeDetectorTest, DetectIvanZolo_HorizontalMotion) {
    MemeDetector detector;

    deque<Point> history = {
        Point(20, 100),
        Point(40, 100),
        Point(60, 100),
        Point(80, 100),
        Point(100, 100)
    };

    // 0 пальцев
    auto result = detector.detect(history, 0);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), MEME_IVANZOLO);
}

// 4. иванзоло + 0
TEST(MemeDetectorTest, IvanZolo_FailsWithWrongFingers) {
    MemeDetector detector;
    deque<Point> history = {
        Point(20, 100), Point(40, 100), Point(60, 100), Point(80, 100), Point(100, 100)
    };


    auto result = detector.detect(history, 5);
    EXPECT_FALSE(result.has_value());
}

// 5.гризман тесты
TEST(MemeDetectorTest, DetectGrizman_StaticMotion) {
    MemeDetector detector;

    deque<Point> history = {
        Point(100, 100),
        Point(105, 102),
        Point(98, 99),
        Point(102, 105),
        Point(100, 100)
    };

    // 2
    auto result2 = detector.detect(history, 2);
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), MEME_GRIZMAN);

    // 3
    auto result3 = detector.detect(history, 3);
    ASSERT_TRUE(result3.has_value());
    EXPECT_EQ(result3.value(), MEME_GRIZMAN);
}

// 6. неверное количество пальцев гризмана
TEST(MemeDetectorTest, Grizman_FailsWithWrongFingers) {
    MemeDetector detector;
    deque<Point> history = {
        Point(100, 100), Point(105, 102), Point(98, 99), Point(102, 105), Point(100, 100)
    };

    auto result = detector.detect(history, 5);
    EXPECT_FALSE(result.has_value());
}