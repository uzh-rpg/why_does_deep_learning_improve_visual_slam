#ifndef OPTICALFLOWLOADER_H
#define OPTICALFLOWLOADER_H

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <H5Cpp.h>

namespace ORB_SLAM3
{

class OpticalFlowLoader
{
public:
    // Constructor: h5FilePath is the path to the HDF5 file containing optical flow data
    OpticalFlowLoader(const std::string &h5FilePath);

    // Get optical flow for a given (srcTimestamp, destTimestamp) pair
    // Returns flow upsampled to full image resolution
    // Returns true if flow exists, false otherwise
    bool GetFlowByTimestamps(int64_t srcTimestamp, int64_t destTimestamp, cv::Mat &flowX, cv::Mat &flowY, bool upsample=true);
    
    // Get optical flow displacement at a specific pixel location (in full image resolution coordinates)
    // The method handles coordinate scaling and bilinear interpolation internally
    // Returns flow values scaled to full image resolution
    // Returns true if valid, false if out of bounds or timestamps not found
    bool GetFlowAtPixel(int64_t srcTimestamp, int64_t destTimestamp, float x, float y, int octave, float &dx, float &dy);

    // Get average weight at a specific pixel location (in full image resolution coordinates)
    // Returns the mean weight across the 2 channels, upsampled via bilinear interpolation
    // Returns true if valid, false if out of bounds or timestamps not found
    bool GetWeightAtPixel(int64_t srcTimestamp, int64_t destTimestamp, float x, float y, int octave, float &weight);

    // Get weight map for a given (srcTimestamp, destTimestamp) pair
    // Returns the average weight per pixel (mean of 2 channels), optionally upsampled to full image resolution
    // Returns true if weights exist, false otherwise
    bool GetWeightsByTimestamps(int64_t srcTimestamp, int64_t destTimestamp, cv::Mat &weights, bool upsample=true);
    
    // Get internal flow index from (src, dest) timestamp pair
    // Returns -1 if not found
    int GetFlowIndex(int64_t srcTimestamp, int64_t destTimestamp);
    
    // Get all destination timestamps available for a given source timestamp
    std::vector<int64_t> GetDestTimestampsForSource(int64_t srcTimestamp);
    
    bool IsLoaded() const { return mbLoaded; }
    
    // Get flow dimensions (at 1/8 resolution)
    int GetFlowHeight() const { return mFlowHeight; }
    int GetFlowWidth() const { return mFlowWidth; }
    
    // Get original image dimensions
    int GetImageHeight() const { return mImageHeight; }
    int GetImageWidth() const { return mImageWidth; }
    
    // Get scale factor (flow is stored at 1/mScaleFactor resolution)
    int GetScaleFactor() const { return mScaleFactor; }

    // 
    cv::Mat bilinearInterpolation(const cv::Mat& input);
    float bilinearInterpolation(const cv::Mat& input, float upSample, int x_out, int y_out);

private:
    // Convert double timestamp to integer (nanoseconds) for reliable comparison
    static int64_t TimestampToNanos(double timestamp);
    
    bool mbLoaded;
    int mNumFlows;          // Number of flow entries
    int mFlowHeight;        // Height of stored flow (1/8 resolution)
    int mFlowWidth;         // Width of stored flow (1/8 resolution)
    int mImageHeight;       // Original image height
    int mImageWidth;        // Original image width
    int mScaleFactor;       // Scale factor (8)
    
    // Store all optical flow data: vector index = flow index, cv::Mat is 2-channel (dx, dy)
    std::vector<cv::Mat> mvOpticalFlow;

    // Store average weights per pixel: vector index = flow index, cv::Mat is single-channel (mean of 2 channels)
    std::vector<cv::Mat> mvWeights;
    
    // Store timestamp pairs
    std::vector<std::pair<int64_t, int64_t>> mvTimestampPairs;

    // Map from (srcTimestamp, destTimestamp) to flow index
    std::map<std::pair<int64_t, int64_t>, int> mmTimestampPairToIndex;
    
    // Map from srcTimestamp to list of destTimestamps
    std::map<int64_t, std::vector<int64_t>> mmSrcToDestTimestamps;
};

} // namespace ORB_SLAM3

#endif // OPTICALFLOWLOADER_H