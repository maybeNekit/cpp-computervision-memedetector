#include "HandState.h"
#include <map>

void HandState::reset() {
    fingerHistory.clear();
    radiusBuffer.clear();
    positionHistory.clear();
    smoothCenter = cv::Point(0, 0);
    smoothRadius = 0;
    isFirstFrame = true;
    displayedFingers = 0;
}

void HandState::updateVoting(int currentFingers) {
    fingerHistory.push_back(currentFingers);
    if (fingerHistory.size() > 10) fingerHistory.pop_front();

    std::map<int, int> votes;
    for (int val : fingerHistory) votes[val]++;
    int maxVotes = 0;
    for (auto const& item : votes) {
        if (item.second > maxVotes) {
            maxVotes = item.second;
            displayedFingers = item.first;
        }
    }
}