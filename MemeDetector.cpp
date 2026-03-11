#include "MemeDetector.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace std;

MemeType MemeDetector::detect(const deque<Point>& history, int fingers) {
    if (history.size() < 5) return MEME_NONE;

    vector<int> x_vals;
    vector<int> y_vals;

    for (const auto& p : history) {
        x_vals.push_back(p.x);
        y_vals.push_back(p.y);
    }

    auto minmax_x = minmax_element(x_vals.begin(), x_vals.end());
    auto minmax_y = minmax_element(y_vals.begin(), y_vals.end());

    double deltaX = *minmax_x.second - *minmax_x.first;
    double deltaY = *minmax_y.second - *minmax_y.first;

    bool isVerticalMotion = deltaY > deltaX * 1.2;
    bool isSignificant = deltaY > 35;

    if (isVerticalMotion && isSignificant) {
        return MEME_67;
    }

    bool isStatic = deltaX < 40 && deltaY < 40;
    bool correctFingers = (fingers == 2 || fingers == 3);

    if (isStatic && correctFingers) {
        return MEME_GRIZMAN;
    }

    return MEME_NONE;
}