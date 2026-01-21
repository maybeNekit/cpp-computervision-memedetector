#include "MemeDetector.h"
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

MemeType MemeDetector::detect(const deque<Point>& history, int fingerCount) {
    if (history.size() < 5) return MEME_NONE;

    if (fingerCount < 3) return MEME_NONE;

    int minY = 10000, maxY = -1;
    int minX = 10000, maxX = -1;

    for (const auto& p : history) {
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
    }

    int diffY = maxY - minY;
    int diffX = maxX - minX;

    if (diffY > 35 && diffX < 50) {
        if (diffY > diffX * 1.2) {
            return MEME_67;
        }
    }

    return MEME_NONE;
}