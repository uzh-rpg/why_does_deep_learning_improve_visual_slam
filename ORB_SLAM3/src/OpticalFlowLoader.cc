#include "OpticalFlowLoader.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <opencv2/imgproc.hpp>

namespace ORB_SLAM3
{

int64_t OpticalFlowLoader::TimestampToNanos(double timestamp)
{
    // Convert seconds to nanoseconds for integer comparison
    return static_cast<int64_t>(std::round(timestamp * 1e9));
}

OpticalFlowLoader::OpticalFlowLoader(const std::string &h5FilePath)
    : mbLoaded(false), mNumFlows(0), mFlowHeight(0), mFlowWidth(0), 
      mImageHeight(0), mImageWidth(0), mScaleFactor(8)
{
    try
    {        
        // Load H5 file
        H5::H5File file(h5FilePath, H5F_ACC_RDONLY);
        
        // Load timestamps dataset (n, 2) - [src_timestamp, dest_timestamp]
        H5::DataSet timestampsDataset = file.openDataSet("timestamps");
        H5::DataSpace timestampsDataspace = timestampsDataset.getSpace();
        
        int tsNdims = timestampsDataspace.getSimpleExtentNdims();
        if (tsNdims != 2)
        {
            std::cerr << "OpticalFlowLoader: Expected 2D timestamps array (n, 2), got " << tsNdims << "D" << std::endl;
            return;
        }
        
        hsize_t tsDims[2];
        timestampsDataspace.getSimpleExtentDims(tsDims);
        mNumFlows = static_cast<int>(tsDims[0]);
        
        if (tsDims[1] != 2)
        {
            std::cerr << "OpticalFlowLoader: Expected timestamps shape (n, 2), got (n, " << tsDims[1] << ")" << std::endl;
            return;
        }
        
        // Read timestamps
        std::vector<int64_t> timestampsBuffer(mNumFlows * 2);
        timestampsDataset.read(timestampsBuffer.data(), H5::PredType::NATIVE_INT64);
        
        // Build timestamp pair mappings using integer nanoseconds for keys
        mvTimestampPairs.resize(mNumFlows);
        for (int i = 0; i < mNumFlows; i++)
        {
            int64_t srcTs = timestampsBuffer[i * 2];
            int64_t destTs = timestampsBuffer[i * 2 + 1];
            mvTimestampPairs[i] = std::make_pair(srcTs, destTs);
            
            // Use nanoseconds for map keys to avoid floating-point comparison issues
            mmTimestampPairToIndex[std::make_pair(srcTs, destTs)] = i;
            mmSrcToDestTimestamps[srcTs].push_back(destTs);
        }
        
        // Load flow dataset (n, h, w, 2)
        H5::DataSet flowDataset = file.openDataSet("flow");
        H5::DataSpace flowDataspace = flowDataset.getSpace();
        
        int flowNdims = flowDataspace.getSimpleExtentNdims();
        if (flowNdims != 4)
        {
            std::cerr << "OpticalFlowLoader: Expected 4D flow array (n, h, w, 2), got " << flowNdims << "D" << std::endl;
            return;
        }
        
        hsize_t flowDims[4];
        flowDataspace.getSimpleExtentDims(flowDims);
        
        if (static_cast<int>(flowDims[0]) != mNumFlows)
        {
            std::cerr << "OpticalFlowLoader: Flow count (" << flowDims[0] 
                      << ") doesn't match timestamps count (" << mNumFlows << ")" << std::endl;
            return;
        }
        
        mFlowHeight = static_cast<int>(flowDims[1]);
        mFlowWidth = static_cast<int>(flowDims[2]);
        int channels = static_cast<int>(flowDims[3]);
        
        if (channels != 2)
        {
            std::cerr << "OpticalFlowLoader: Expected 2 channels (dx, dy), got " << channels << std::endl;
            return;
        }
        
        // Compute original image dimensions
        mImageHeight = mFlowHeight * mScaleFactor;
        mImageWidth = mFlowWidth * mScaleFactor;
        
        std::cout << "OpticalFlowLoader: Loading " << mNumFlows << " flow entries of size (width x height): " 
                  << mFlowWidth << "x" << mFlowHeight << " (1/" << mScaleFactor << " resolution)" << std::endl;
        std::cout << "OpticalFlowLoader: Original image size (width x height): " << mImageWidth << "x" << mImageHeight << std::endl;
        
        // Read all flow data into a buffer
        std::vector<float> buffer(mNumFlows * mFlowHeight * mFlowWidth * 2);
        flowDataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);
        
        // Convert to vector of cv::Mat
        mvOpticalFlow.resize(mNumFlows);
        const size_t frameBytes = mFlowHeight * mFlowWidth * 2 * sizeof(float);
        for (int i = 0; i < mNumFlows; i++)
        {
            mvOpticalFlow[i] = cv::Mat(mFlowHeight, mFlowWidth, CV_32FC2);
            std::memcpy(mvOpticalFlow[i].data, 
                        buffer.data() + static_cast<size_t>(i) * mFlowHeight * mFlowWidth * 2,
                        frameBytes);
        }

        // Load weights dataset (n, h, w, 2)
        H5::DataSet weightsDataset = file.openDataSet("weights");
        H5::DataSpace weightsDataspace = weightsDataset.getSpace();
        
        int weightsNdims = weightsDataspace.getSimpleExtentNdims();
        if (weightsNdims != 4)
        {
            std::cerr << "OpticalFlowLoader: Expected 4D weights array (n, h, w, 2), got " << weightsNdims << "D" << std::endl;
            return;
        }
        
        hsize_t weightsDims[4];
        weightsDataspace.getSimpleExtentDims(weightsDims);
        
        if (static_cast<int>(weightsDims[0]) != mNumFlows ||
            static_cast<int>(weightsDims[1]) != mFlowHeight ||
            static_cast<int>(weightsDims[2]) != mFlowWidth ||
            static_cast<int>(weightsDims[3]) != 2)
        {
            std::cerr << "OpticalFlowLoader: Weights shape mismatch. Expected (" 
                      << mNumFlows << ", " << mFlowHeight << ", " << mFlowWidth << ", 2), got ("
                      << weightsDims[0] << ", " << weightsDims[1] << ", " << weightsDims[2] << ", " << weightsDims[3] << ")" << std::endl;
            return;
        }
        
        // Read all weights data into a buffer
        std::vector<float> weightsBuffer(mNumFlows * mFlowHeight * mFlowWidth * 2);
        weightsDataset.read(weightsBuffer.data(), H5::PredType::NATIVE_FLOAT);
        
        // Convert to vector of cv::Mat, averaging over the 2 channels to get per-pixel weight
        mvWeights.resize(mNumFlows);
        for (int i = 0; i < mNumFlows; i++)
        {
            mvWeights[i] = cv::Mat(mFlowHeight, mFlowWidth, CV_32FC1);
            const float* src = weightsBuffer.data() + static_cast<size_t>(i) * mFlowHeight * mFlowWidth * 2;
            float* dst = reinterpret_cast<float*>(mvWeights[i].data);
            for (int j = 0; j < mFlowHeight * mFlowWidth; j++)
            {
                // Average over the 2 channels: weights = mean(weights[:,:,0], weights[:,:,1])
                dst[j] = (src[j * 2] + src[j * 2 + 1]) * 0.5f;
            }
        }
        
        file.close();
        mbLoaded = true;
        std::cout << "OpticalFlowLoader: Successfully loaded optical flow and weights data" << std::endl;
    }
    catch (H5::Exception &e)
    {
        std::cerr << "OpticalFlowLoader: HDF5 error: " << e.getCDetailMsg() << std::endl;
    }
    catch (std::exception &e)
    {
        std::cerr << "OpticalFlowLoader: Error: " << e.what() << std::endl;
    }
}

int OpticalFlowLoader::GetFlowIndex(int64_t srcTimestamp, int64_t destTimestamp)
{
    auto key = std::make_pair(srcTimestamp, destTimestamp);

    auto it = mmTimestampPairToIndex.find(key);
    if (it != mmTimestampPairToIndex.end())
        return it->second;
    return -1;
}

std::vector<int64_t> OpticalFlowLoader::GetDestTimestampsForSource(int64_t srcTimestamp)
{
    auto it = mmSrcToDestTimestamps.find(srcTimestamp);
    if (it != mmSrcToDestTimestamps.end())
        return it->second;
    return std::vector<int64_t>();
}

bool OpticalFlowLoader::GetFlowByTimestamps(int64_t srcTimestamp, int64_t destTimestamp, cv::Mat &flowX, cv::Mat &flowY, bool upsample)
{
    int flowIndex = GetFlowIndex(srcTimestamp, destTimestamp);
    if (!mbLoaded || flowIndex < 0 || flowIndex >= mNumFlows)
        return false;
    
    // Split into u and v channels
    cv::Mat channels[2];
    cv::split(mvOpticalFlow[flowIndex], channels);
    
    if (upsample)
    {
        // Upsample using bilinear interpolation (cv::resize uses INTER_LINEAR by default)
        cv::Mat upsampledX, upsampledY;

        // opencv resize
        cv::resize(channels[0], upsampledX, cv::Size(mImageWidth, mImageHeight), 0, 0, cv::INTER_LINEAR);
        cv::resize(channels[1], upsampledY, cv::Size(mImageWidth, mImageHeight), 0, 0, cv::INTER_LINEAR);

        // custom bilinear interpolation
        // upsampledX = bilinearInterpolation(channels[0]);
        // upsampledY = bilinearInterpolation(channels[1]);

        float scale = static_cast<float>(mScaleFactor);
        
        // Scale flow values by scale factor (flow was computed at 1/8 resolution)
        flowX = upsampledX * scale;
        flowY = upsampledY * scale;
    }
    else
    {
        flowX = channels[0];
        flowY = channels[1];
    }
    
    return true;
}

bool OpticalFlowLoader::GetWeightsByTimestamps(int64_t srcTimestamp, int64_t destTimestamp, cv::Mat &weights, bool upsample)
{
    int flowIndex = GetFlowIndex(srcTimestamp, destTimestamp);
    if (!mbLoaded || flowIndex < 0 || flowIndex >= mNumFlows)
        return false;
    
    if (upsample)
    {
        cv::resize(mvWeights[flowIndex], weights, cv::Size(mImageWidth, mImageHeight), 0, 0, cv::INTER_LINEAR);
    }
    else
    {
        weights = mvWeights[flowIndex];
    }
    
    return true;
}

bool OpticalFlowLoader::GetFlowAtPixel(
    int64_t srcTimestamp, int64_t destTimestamp, float x, float y, int octave, float &dx, float &dy)
{
    int flowIndex = GetFlowIndex(srcTimestamp, destTimestamp);
    if (!mbLoaded || flowIndex < 0 || flowIndex >= mNumFlows)
        return false;

    // Use bilinear interpolation to get flow at (x, y) in full image resolution
    // Split into u and v channels
    cv::Mat channels[2];
    cv::split(mvOpticalFlow[flowIndex], channels);

    float upSample = static_cast<float>(mScaleFactor);
    for (int i = 0; i < octave; i++)
    {
        // upSample = upSample / 2.0f;

        // use if ORBextractor.scaleFactor: 1.2f
        upSample = upSample / 1.2f;
    }

    dx = bilinearInterpolation(channels[0], upSample, static_cast<int>(x), static_cast<int>(y));
    dx *= upSample;
    dy = bilinearInterpolation(channels[1], upSample, static_cast<int>(x), static_cast<int>(y));
    dy *= upSample;
    
    return true;
}

bool OpticalFlowLoader::GetWeightAtPixel(
    int64_t srcTimestamp, int64_t destTimestamp, float x, float y, int octave, float &weight)
{
    int flowIndex = GetFlowIndex(srcTimestamp, destTimestamp);
    if (!mbLoaded || flowIndex < 0 || flowIndex >= mNumFlows)
        return false;

    float upSample = static_cast<float>(mScaleFactor);
    for (int i = 0; i < octave; i++)
    {
        // use if ORBextractor.scaleFactor: 1.2f
        upSample = upSample / 1.2f;
    }

    weight = bilinearInterpolation(mvWeights[flowIndex], upSample, static_cast<int>(x), static_cast<int>(y));
    
    return true;
}

cv::Mat OpticalFlowLoader::bilinearInterpolation(const cv::Mat& input)
{
    int zoom = mScaleFactor;

    int inH = input.rows;
    int inW = input.cols;
    int outH = static_cast<int>(inH * zoom + 0.5f);
    int outW = static_cast<int>(inW * zoom + 0.5f);

    cv::Mat output(outH, outW, CV_32FC1);

    for (int y = 0; y < outH; ++y)
    {
        float inY = (y + 0.5f) / zoom - 0.5f;
        int y0 = static_cast<int>(std::floor(inY));
        int y1 = y0 + 1;
        float wy = inY - y0;

        y0 = std::max(0, std::min(y0, inH - 1));
        y1 = std::max(0, std::min(y1, inH - 1));

        const float* row0 = input.ptr<float>(y0);
        const float* row1 = input.ptr<float>(y1);
        float* outRow = output.ptr<float>(y);

        for (int x = 0; x < outW; ++x)
        {
            float inX = (x + 0.5f) / zoom - 0.5f;
            int x0 = static_cast<int>(std::floor(inX));
            int x1 = x0 + 1;
            float wx = inX - x0;

            x0 = std::max(0, std::min(x0, inW - 1));
            x1 = std::max(0, std::min(x1, inW - 1));

            float v00 = row0[x0];
            float v01 = row0[x1];
            float v10 = row1[x0];
            float v11 = row1[x1];

            outRow[x] = (1 - wy) * ((1 - wx) * v00 + wx * v01) +
                        wy       * ((1 - wx) * v10 + wx * v11);
        }
    }

    return output;
}

float OpticalFlowLoader::bilinearInterpolation(const cv::Mat& input, float upSample, int x_out, int y_out)
{
    int H = input.rows;
    int W = input.cols;

    float x_in = (x_out + 0.5f)/upSample - 0.5f;
    float y_in = (y_out + 0.5f)/upSample - 0.5f;

    int x0 = static_cast<int>(std::floor(x_in));
    int x1 = x0 + 1;
    int y0 = static_cast<int>(std::floor(y_in));
    int y1 = y0 + 1;

    float wx = x_in - x0;
    float wy = y_in - y0;

    // clamp
    x0 = std::max(0, std::min(x0, W-1));
    x1 = std::max(0, std::min(x1, W-1));
    y0 = std::max(0, std::min(y0, H-1));
    y1 = std::max(0, std::min(y1, H-1));

    float v00 = input.at<float>(y0, x0);
    float v01 = input.at<float>(y0, x1);
    float v10 = input.at<float>(y1, x0);
    float v11 = input.at<float>(y1, x1);

    return (1-wy)*((1-wx)*v00 + wx*v01) + wy*((1-wx)*v10 + wx*v11);
}

} // namespace ORB_SLAM3