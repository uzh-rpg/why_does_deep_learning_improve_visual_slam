/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"

#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>

namespace ORB_SLAM3
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
{
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
}


//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mTimeStampNs(frame.mTimeStampNs), mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
     mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
     mTcw(frame.mTcw), mbHasPose(false), mbHasVelocity(false), flowWeights(frame.flowWeights)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0){
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(frame.mbHasPose)
        SetPose(frame.GetPose());

    if(frame.HasVelocity())
    {
        SetVelocity(frame.GetVelocity());
    }

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mmProjectPoints.clear();
    mmMatchedInImage.clear();


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);



        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    for (int i = 0; i < N; i++)
    {
        flowWeights.push_back(-1.0f);
    }
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,0);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF){
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else{
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    for (int i = 0; i < N; i++)
    {
        flowWeights.push_back(-1.0f);
    }
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, const int64_t &timeStampNs, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mTimeStampNs(timeStampNs), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Store grayscale image for dynamic keypoint descriptor computation
    mImGray = imGray.clone();

    // Initialize umax table for IC_Angle computation
    const int HALF_PATCH_SIZE = 15;
    mumax.resize(HALF_PATCH_SIZE + 1);
    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        mumax[v] = cvRound(sqrt(hp2 - v * v));
    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (mumax[v0] == mumax[v0 + 1])
            ++v0;
        mumax[v] = v0;
        ++v0;
    }

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,1000);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    for (int i = 0; i < N; i++)
    {
        flowWeights.push_back(-1.0f);
    }
}


void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }



    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft)
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    if(flag==0)
        monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
    else
        monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
}

bool Frame::isSet() const {
    return mbIsSet;
}

void Frame::SetPose(const Sophus::SE3<float> &Tcw) {
    mTcw = Tcw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const
{
    return mVw;
}

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();
}

Eigen::Matrix<float,3,1> Frame::GetImuPosition() const {
    return mRwc * mImuCalib.mTcb.translation() + mOw;
}

Eigen::Matrix<float,3,3> Frame::GetImuRotation() {
    return mRwc * mImuCalib.mTcb.rotationMatrix();
}

Sophus::SE3<float> Frame::GetImuPose() {
    return mTcw.inverse() * mImuCalib.mTcb;
}

Sophus::SE3f Frame::GetRelativePoseTrl()
{
    return mTrl;
}

Sophus::SE3f Frame::GetRelativePoseTlr()
{
    return mTlr;
}

Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation(){
    return mTlr.rotationMatrix();
}

Eigen::Vector3f Frame::GetRelativePoseTlr_translation() {
    return mTlr.translation();
}


bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    if(Nleft == -1){
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Eigen::Matrix<float,3,1> Pc = mRcw * P + mtcw;
        const float Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ = Pc(2);
        const float invz = 1.0f/PcZ;
        if(PcZ<0.0f)
            return false;

        const Eigen::Vector2f uv = mpCamera->project(Pc);

        if(uv(0)<mnMinX || uv(0)>mnMaxX)
            return false;
        if(uv(1)<mnMinY || uv(1)>mnMaxY)
            return false;

        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const Eigen::Vector3f PO = P - mOw;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            return false;

        // Check viewing angle
        Eigen::Vector3f Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjXR = uv(0) - mbf*invz;

        pMP->mTrackDepth = Pc_dist;

        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }
    else{
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mRcw * P + mtcw;
    const float &PcX = Pc(0);
    const float &PcY= Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw)
{
    return mRcw * pCw + mtcw;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

// Constants for ORB descriptor computation
static const int FRAME_PATCH_SIZE = 31;
static const int FRAME_HALF_PATCH_SIZE = 15;
static const int FRAME_EDGE_THRESHOLD = 19;
static const float FRAME_FACTOR_PI = (float)(CV_PI/180.f);

// BRIEF pattern (same as in ORBextractor.cc)
static int frame_bit_pattern_31_[256*4] =
{
    8,-3, 9,5, 4,2, 7,-12, -11,9, -8,2, 7,-12, 12,-13,
    2,-13, 2,12, 1,-7, 1,6, -2,-10, -2,-4, -13,-13, -11,-8,
    -13,-3, -12,-9, 10,4, 11,9, -13,-8, -8,-9, -11,7, -9,12,
    7,7, 12,6, -4,-5, -3,0, -13,2, -12,-3, -9,0, -7,5,
    12,-6, 12,-1, -3,6, -2,12, -6,-13, -4,-8, 11,-13, 12,-8,
    4,7, 5,1, 5,-3, 10,-3, 3,-7, 6,12, -8,-7, -6,-2,
    -2,11, -1,-10, -13,12, -8,10, -7,3, -5,-3, -4,2, -3,7,
    -10,-12, -6,11, 5,-12, 6,-7, 5,-6, 7,-1, 1,0, 4,-5,
    9,11, 11,-13, 4,7, 4,12, 2,-1, 4,4, -4,-12, -2,7,
    -8,-5, -7,-10, 4,11, 9,12, 0,-8, 1,-13, -13,-2, -8,2,
    -3,-2, -2,3, -6,9, -4,-9, 8,12, 10,7, 0,9, 1,3,
    7,-5, 11,-10, -13,-6, -11,0, 10,7, 12,1, -6,-3, -6,12,
    10,-9, 12,-4, -13,8, -8,-12, -13,0, -8,-4, 3,3, 7,8,
    5,7, 10,-7, -1,7, 1,-12, 3,-10, 5,6, 2,-4, 3,-10,
    -13,0, -13,5, -13,-7, -12,12, -13,3, -11,8, -7,12, -4,7,
    6,-10, 12,8, -9,-1, -7,-6, -2,-5, 0,12, -12,5, -7,5,
    3,-10, 8,-13, -7,-7, -4,5, -3,-2, -1,-7, 2,9, 5,-11,
    -11,-13, -5,-13, -1,6, 0,-1, 5,-3, 5,2, -4,-13, -4,12,
    -9,-6, -9,6, -12,-10, -8,-4, 10,2, 12,-3, 7,12, 12,12,
    -7,-13, -6,5, -4,9, -3,4, 7,-1, 12,2, -7,6, -5,1,
    -13,11, -12,5, -3,7, -2,-6, 7,-8, 12,-7, -13,-7, -11,-12,
    1,-3, 12,12, 2,-6, 3,0, -4,3, -2,-13, -1,-13, 1,9,
    7,1, 8,-6, 1,-1, 3,12, 9,1, 12,6, -1,-9, -1,3,
    -13,-13, -10,5, 7,7, 10,12, 12,-5, 12,9, 6,3, 7,11,
    5,-13, 6,10, 2,-12, 2,3, 3,8, 4,-6, 2,6, 12,-13,
    9,-12, 10,3, -8,4, -7,9, -11,12, -4,-6, 1,12, 2,-8,
    6,-9, 7,-4, 2,3, 3,-2, 6,3, 11,0, 3,-3, 8,-8,
    7,8, 9,3, -11,-5, -6,-4, -10,11, -5,10, -5,-8, -3,12,
    -10,5, -9,0, 8,-1, 12,-6, 4,-6, 6,-11, -10,12, -8,7,
    4,-2, 6,7, -2,0, -2,12, -5,-8, -5,2, 7,-6, 10,12,
    -9,-13, -8,-8, -5,-13, -5,-2, 8,-8, 9,-13, -9,-11, -9,0,
    1,-8, 1,-2, 7,-4, 9,1, -2,1, -1,-4, 11,-6, 12,-11,
    -12,-9, -6,4, 3,7, 7,12, 5,5, 10,8, 0,-4, 2,8,
    -9,12, -5,-13, 0,7, 2,12, -1,2, 1,7, 5,11, 7,-9,
    3,5, 6,-8, -13,-4, -8,9, -5,9, -3,-3, -4,-7, -3,-12,
    6,5, 8,0, -7,6, -6,12, -13,6, -5,-2, 1,-10, 3,10,
    4,1, 8,-4, -2,-2, 2,-13, 2,-12, 12,12, -2,-13, 0,-6,
    4,1, 9,3, -6,-10, -3,-5, -3,-13, -1,1, 7,5, 12,-11,
    4,-2, 5,-7, -13,9, -9,-5, 7,1, 8,6, 7,-8, 7,6,
    -7,-4, -7,1, -8,11, -7,-8, -13,6, -12,-8, 2,4, 3,9,
    10,-5, 12,3, -6,-5, -6,7, 8,-3, 9,-8, 2,-12, 2,8,
    -11,-2, -10,3, -12,-13, -7,-9, -11,0, -10,-5, 5,-3, 11,8,
    -2,-13, -1,12, -1,-8, 0,9, -13,-11, -12,-5, -10,-2, -10,11,
    -3,9, -2,-13, 2,-3, 3,2, -9,-13, -4,0, -4,6, -3,-10,
    -4,12, -2,-7, -6,-11, -4,9, 6,-3, 6,11, -13,11, -5,5,
    11,11, 12,6, 7,-5, 12,-2, -1,12, 0,7, -4,-8, -3,-2,
    -7,1, -6,7, -13,-12, -8,-13, -7,-2, -6,-8, -8,5, -6,-9,
    -5,-1, -4,5, -13,7, -8,10, 1,5, 5,-13, 1,0, 10,-13,
    9,12, 10,-1, 5,-8, 10,-9, -1,11, 1,-13, -9,-3, -6,2,
    -1,-10, 1,12, -13,1, -8,-10, 8,-11, 10,-6, 2,-13, 3,-6,
    7,-13, 12,-9, -10,-10, -5,-7, -10,-8, -8,-13, 4,-6, 8,5,
    3,12, 8,-13, -4,2, -3,-3, 5,-13, 10,-12, 4,-13, 5,-1,
    -9,9, -4,3, 0,3, 3,-9, -12,1, -6,1, 3,2, 4,-8,
    -10,-10, -10,9, 8,-13, 12,12, -8,-12, -6,-5, 2,2, 3,7,
    10,6, 11,-8, 6,8, 8,-12, -7,10, -6,5, -3,-9, -3,9,
    -1,-13, -1,5, -3,-7, -3,4, -8,-2, -8,3, 4,2, 12,12,
    2,-5, 3,11, 6,-9, 11,-13, 3,-1, 7,12, 11,-1, 12,4,
    -3,0, -3,6, 4,-11, 4,12, 2,-4, 2,1, -10,-6, -8,1,
    -13,7, -11,1, -13,12, -11,-13, 6,0, 11,-13, 0,-1, 1,4,
    -13,3, -9,-2, -9,8, -6,-3, -13,-6, -8,-2, 5,-9, 8,10,
    2,7, 3,-9, -1,-6, -1,-1, 9,5, 11,-2, 11,-3, 12,-8,
    3,0, 3,5, -1,4, 0,10, 3,-6, 4,5, -13,0, -10,5,
    5,8, 12,11, 8,9, 9,-6, 7,-4, 8,-12, -10,4, -10,9,
    7,3, 12,4, 9,-7, 10,-2, 7,0, 12,-2, -1,-6, 0,-11
};

// Compute IC_Angle (same algorithm as ORBextractor.cc)
static float Frame_IC_Angle(const cv::Mat& image, cv::Point2f pt, const std::vector<int>& u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -FRAME_HALF_PATCH_SIZE; u <= FRAME_HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= FRAME_HALF_PATCH_SIZE; ++v)
    {
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return cv::fastAtan2((float)m_01, (float)m_10);
}

// Compute ORB descriptor (same algorithm as ORBextractor.cc)
static void Frame_computeOrbDescriptor(const cv::KeyPoint& kpt, const cv::Mat& img, 
                                        const cv::Point* pattern, uchar* desc)
{
    float angle = (float)kpt.angle * FRAME_FACTOR_PI;
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]

    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

    #undef GET_VALUE
}

int Frame::AddKeyPoint(int x, int y, int octave, float size, MapPoint* pMP)
{
    // Validate octave is within bounds
    if (octave < 0 || octave >= static_cast<int>(mvScaleFactors.size()))
    {
        std::cerr << "AddKeyPoint: Invalid octave " << octave << std::endl;
        return -1;
    }

    // Check if keypoint is far enough from border for descriptor computation
    // Border size is EDGE_THRESHOLD (19) at octave 0, scaled for higher octaves
    const int borderSize = FRAME_EDGE_THRESHOLD + 3;  // Add extra margin for safety
    if (x < borderSize || x >= mImGray.cols - borderSize ||
        y < borderSize || y >= mImGray.rows - borderSize)
    {
        // Too close to border, cannot compute descriptor
        return -1;
    }

    // Create the new keypoint
    cv::KeyPoint newKp;
    newKp.pt.x = x;
    newKp.pt.y = y;
    newKp.octave = octave;
    newKp.size = size;

    // Compute keypoint angle using intensity centroid method
    newKp.angle = Frame_IC_Angle(mImGray, newKp.pt, mumax);

    // Compute ORB descriptor
    cv::Mat descriptor(1, 32, CV_8U);
    const cv::Point* pattern = (const cv::Point*)frame_bit_pattern_31_;
    Frame_computeOrbDescriptor(newKp, mImGray, pattern, descriptor.ptr<uchar>());

    // Get the new index
    int newIdx = N;

    // Add to keypoint vectors
    mvKeysUn.push_back(newKp);
    mvKeys.push_back(newKp);  // For monocular, mvKeys and mvKeysUn are similar

    // Add MapPoint association
    mvpMapPoints.push_back(pMP);

    // Add outlier flag (not an outlier initially)
    mvbOutlier.push_back(false);

    // Add stereo info (monocular case: -1)
    mvuRight.push_back(-1.0f);
    mvDepth.push_back(-1.0f);

    // Add the computed descriptor
    mDescriptors.push_back(descriptor);

    // Add to grid for GetFeaturesInArea to work
    int nGridPosX, nGridPosY;
    if(PosInGrid(newKp, nGridPosX, nGridPosY))
    {
        mGrid[nGridPosX][nGridPosY].push_back(newIdx);
    }

    // Increment keypoint count
    N++;

    return newIdx;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);

    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if(z>0) {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = mRwc * x3Dc + mOw;
        return true;
    } else
        return false;
}

bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),  mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2),
         mbHasPose(false), mbHasVelocity(false)

{
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    // Sophus/Eigen
    mTlr = Tlr;
    mTrl = mTlr.inverse();
    mRlr = mTlr.rotationMatrix();
    mtlr = mTlr.translation();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();

}

void Frame::ComputeStereoFishEyeMatches() {
    //Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs = 0;

    //Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    //Check matches using Lowe's ratio
    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
            //For every good match, check parallax and reprojection error to discard spurious matches
            Eigen::Vector3f p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],mvKeysRight[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
            if(depth > 0.0001f){
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D;
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if(bRight){
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.translation() + mOw;
    }
    else{
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mR * P + mt;
    const float Pc_dist = Pc.norm();
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    if(bRight) uv = mpCamera2->project(Pc);
    else uv = mpCamera->project(Pc);

    if(uv(0)<mnMinX || uv(0)>mnMaxX)
        return false;
    if(uv(1)<mnMinY || uv(1)>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = P - twc;
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    Eigen::Vector3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    if(bRight){
        pMP->mTrackProjXR = uv(0);
        pMP->mTrackProjYR = uv(1);
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else{
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i){
    return mRwc * mvStereo3Dpoints[i] + mOw;
}

} //namespace ORB_SLAM
