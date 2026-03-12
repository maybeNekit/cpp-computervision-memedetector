#include <gtest/gtest.h>
#include "MemeDetector.h"

TEST(MemeDetectorTest, EmptyHistoryReturnsNullopt) {
    MemeDetector detector;
    std::deque<cv::Point> emptyHistory;

    auto result = detector.detect(emptyHistory, 0);

    EXPECT_FALSE(result.has_value());
}