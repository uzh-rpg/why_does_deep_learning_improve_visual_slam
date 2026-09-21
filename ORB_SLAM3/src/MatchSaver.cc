#include "MatchSaver.h"

#include <iostream>

namespace ORB_SLAM3 {

MatchSaver::MatchSaver(const std::string& filename, int width, int height)
    : mWidth(width), mHeight(height)
{
    mFile.open(filename, std::ios::binary);
    if (!mFile.is_open()) {
        std::cerr << "Error: Could not open match file: " << filename << std::endl;
        return;
    }
    // Write header: width, height
    mFile.write(reinterpret_cast<const char*>(&mWidth), sizeof(int));
    mFile.write(reinterpret_cast<const char*>(&mHeight), sizeof(int));
}

MatchSaver::~MatchSaver() {
    Close();
}

void MatchSaver::SaveFrameMatches(int64_t lastTimestamp, int64_t currentTimestamp,
                                   const std::vector<cv::KeyPoint>& lastKeypoints,
                                   const std::vector<cv::KeyPoint>& currentKeypoints,
                                   const std::vector<std::pair<int, int>>& matches)
{
    if (!mFile.is_open()) return;
    
    // Create HxWx2 matrix initialized to -1 (no match)
    std::vector<float> matchMatrix(mHeight * mWidth * 2, -1.0f);
    
    for (const auto& match : matches) {
        int lastIdx = match.first;
        int currentIdx = match.second;
        
        const cv::KeyPoint& lastKp = lastKeypoints[lastIdx];
        const cv::KeyPoint& currentKp = currentKeypoints[currentIdx];
        
        int u = static_cast<int>(std::round(lastKp.pt.x));
        int v = static_cast<int>(std::round(lastKp.pt.y));
        
        if (u >= 0 && u < mWidth && v >= 0 && v < mHeight) {
            int idx = (v * mWidth + u) * 2;
            matchMatrix[idx] = currentKp.pt.x;     // target u
            matchMatrix[idx + 1] = currentKp.pt.y; // target v
        }
    }
    
    // Write timestamp and matrix
    mFile.write(reinterpret_cast<const char*>(&lastTimestamp), sizeof(int64_t));
    mFile.write(reinterpret_cast<const char*>(&currentTimestamp), sizeof(int64_t));
    mFile.write(reinterpret_cast<const char*>(matchMatrix.data()), 
                matchMatrix.size() * sizeof(float));
}

// Save dense optical flow for debugging upsampling
void MatchSaver::SaveDenseFlow(int64_t lastTimestamp, int64_t currentTimestamp, cv::Mat flowU, cv::Mat flowV)
{
    if (!mFile.is_open()) return;
    
    // Create HxWx2 matrix initialized to -1 (no match)
    std::vector<float> flowMatrix(mHeight * mWidth * 2, -1.0f);

    for(int i = 0; i < mHeight; i++) 
    {
        for(int j = 0; j < mWidth; j++) 
        {
            float u_flow = flowU.at<float>(i, j);
            float v_flow = flowV.at<float>(i, j);
            int idx = (i * mWidth + j) * 2;
            flowMatrix[idx] = u_flow;
            flowMatrix[idx + 1] = v_flow;
        }
    }
    
    // Write timestamp and matrix
    mFile.write(reinterpret_cast<const char*>(&lastTimestamp), sizeof(int64_t));
    mFile.write(reinterpret_cast<const char*>(&currentTimestamp), sizeof(int64_t));
    mFile.write(reinterpret_cast<const char*>(flowMatrix.data()), 
                flowMatrix.size() * sizeof(float));
}

void MatchSaver::Close() {
    if (mFile.is_open()) {
        mFile.close();
    }
}

} // namespace ORB_SLAM3