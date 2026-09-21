#ifndef MATCHSAVER_H
#define MATCHSAVER_H

#include <string>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>

namespace ORB_SLAM3 {

class MatchSaver {
public:
    MatchSaver(const std::string& filename, int width, int height);
    ~MatchSaver();
    
    // Save matches for a frame: keypoints from LastFrame matched to CurrentFrame
    void SaveFrameMatches(int64_t lastTimestamp, int64_t currentTimestamp,
                          const std::vector<cv::KeyPoint>& lastKeypoints,
                          const std::vector<cv::KeyPoint>& currentKeypoints,
                          const std::vector<std::pair<int, int>>& matches); // pairs of (lastIdx, currentIdx)

    void SaveDenseFlow(int64_t lastTimestamp, int64_t currentTimestamp, cv::Mat flowU, cv::Mat flowV);
    
    void Close();

private:
    std::ofstream mFile;
    int mWidth;
    int mHeight;
};

} // namespace ORB_SLAM3

#endif