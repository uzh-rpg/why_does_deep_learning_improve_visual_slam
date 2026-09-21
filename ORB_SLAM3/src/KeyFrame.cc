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

#include "KeyFrame.h"
#include "Converter.h"
#include "ImuTypes.h"
#include<mutex>

namespace ORB_SLAM3
{

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame():
        mnFrameId(0),  mTimeStamp(0), mTimeStampNs(0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
        mfGridElementWidthInv(0), mfGridElementHeightInv(0),
        mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0),
        mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnMergeQuery(0), mnMergeWords(0), mnBAGlobalForKF(0),
        fx(0), fy(0), cx(0), cy(0), invfx(0), invfy(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
        mbf(0), mb(0), mThDepth(0), N(0), mvKeys(static_cast<vector<cv::KeyPoint> >(NULL)), mvKeysUn(static_cast<vector<cv::KeyPoint> >(NULL)),
        mvuRight(static_cast<vector<float> >(NULL)), mvDepth(static_cast<vector<float> >(NULL)), mnScaleLevels(0), mfScaleFactor(0),
        mfLogScaleFactor(0), mvScaleFactors(0), mvLevelSigma2(0), mvInvLevelSigma2(0), mnMinX(0), mnMinY(0), mnMaxX(0),
        mnMaxY(0), mPrevKF(static_cast<KeyFrame*>(NULL)), mNextKF(static_cast<KeyFrame*>(NULL)), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
        mbToBeErased(false), mbBad(false), mHalfBaseline(0), mbCurrentPlaceRecognition(false), mnMergeCorrectedForKF(0),
        NLeft(0),NRight(0), mnNumberOfOpt(0), mbHasVelocity(false)
{

}

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    bImu(pMap->isImuInitialized()), mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mTimeStampNs(F.mTimeStampNs), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK_(F.mK_), mPrevKF(NULL), mNextKF(NULL), mpImuPreintegrated(F.mpImuPreintegrated),
    mImuCalib(F.mImuCalib), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mDistCoef(F.mDistCoef), mbNotErase(false), mnDataset(F.mnDataset),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap), mbCurrentPlaceRecognition(false), mNameFile(F.mNameFile), mnMergeCorrectedForKF(0),
    mpCamera(F.mpCamera), mpCamera2(F.mpCamera2),
    mvLeftToRightMatch(F.mvLeftToRightMatch),mvRightToLeftMatch(F.mvRightToLeftMatch), mTlr(F.GetRelativePoseTlr()),
    mvKeysRight(F.mvKeysRight), NLeft(F.Nleft), NRight(F.Nright), mTrl(F.GetRelativePoseTrl()), mnNumberOfOpt(0), mbHasVelocity(false)
{
    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    if(F.Nleft != -1)  mGridRight.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        if(F.Nleft != -1) mGridRight[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++){
            mGrid[i][j] = F.mGrid[i][j];
            if(F.Nleft != -1){
                mGridRight[i][j] = F.mGridRight[i][j];
            }
        }
    }



    if(!F.HasVelocity()) {
        mVw.setZero();
        mbHasVelocity = false;
    }
    else
    {
        mVw = F.GetVelocity();
        mbHasVelocity = true;
    }

    mImuBias = F.mImuBias;
    SetPose(F.GetPose());

    mnOriginMapId = pMap->GetId();

    mImGray = F.mImGray.clone();

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
}

void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexPose);

    mTcw = Tcw;
    mRcw = mTcw.rotationMatrix();
    mTwc = mTcw.inverse();
    mRwc = mTwc.rotationMatrix();

    if (mImuCalib.mbIsSet) // TODO Use a flag instead of the OpenCV matrix
    {
        mOwb = mRwc * mImuCalib.mTcb.translation() + mTwc.translation();
    }
}

void KeyFrame::SetVelocity(const Eigen::Vector3f &Vw)
{
    unique_lock<mutex> lock(mMutexPose);
    mVw = Vw;
    mbHasVelocity = true;
}

Sophus::SE3f KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw;
}

Sophus::SE3f KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc;
}

Eigen::Vector3f KeyFrame::GetCameraCenter(){
    unique_lock<mutex> lock(mMutexPose);
    return mTwc.translation();
}

Eigen::Vector3f KeyFrame::GetImuPosition()
{
    unique_lock<mutex> lock(mMutexPose);
    return mOwb;
}

Eigen::Matrix3f KeyFrame::GetImuRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return (mTwc * mImuCalib.mTcb).rotationMatrix();
}

Sophus::SE3f KeyFrame::GetImuPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc * mImuCalib.mTcb;
}

Eigen::Matrix3f KeyFrame::GetRotation(){
    unique_lock<mutex> lock(mMutexPose);
    return mRcw;
}

Eigen::Vector3f KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw.translation();
}

Eigen::Vector3f KeyFrame::GetVelocity()
{
    unique_lock<mutex> lock(mMutexPose);
    return mVw;
}

bool KeyFrame::isVelocitySet()
{
    unique_lock<mutex> lock(mMutexPose);
    return mbHasVelocity;
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        if(!vPairs[i].second->isBad())
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
    {
        return vector<KeyFrame*>();
    }

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);

    if(it==mvOrderedWeights.end() && mvOrderedWeights.back() < w)
    {
        return vector<KeyFrame*>();
    }
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

int KeyFrame::GetNumberMPs()
{
    unique_lock<mutex> lock(mMutexFeatures);
    int numberMPs = 0;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        numberMPs++;
    }
    return numberMPs;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const int &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    tuple<size_t,size_t> indexes = pMP->GetIndexInKeyFrame(this);
    size_t leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    if(leftIndex != -1)
        mvpMapPoints[leftIndex]=static_cast<MapPoint*>(NULL);
    if(rightIndex != -1)
        mvpMapPoints[rightIndex]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const int &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections(bool upParent)
{
    map<KeyFrame*,int> KFcounter;

    vector<MapPoint*> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId || mit->first->isBad() || mit->first->GetMap() != mpMap)
                continue;
            KFcounter[mit->first]++;

        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    
    int th = 8; // 15

    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    if(!upParent)
        cout << "UPDATE_CONN: current KF " << mnId << endl;
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(!upParent)
            cout << "  UPDATE_CONN: KF " << mit->first->mnId << " ; num matches: " << mit->second << endl;
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());


        if(mbFirstConnection && mnId!=mpMap->GetInitKFid())
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    if(pKF == this)
    {
        cout << "ERROR: Change parent KF, the parent and child are the same KF" << endl;
        throw std::invalid_argument("The parent and child can not be the same");
    }

    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::SetFirstConnection(bool bFirst)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbFirstConnection=bFirst;
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::AddMergeEdge(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspMergeEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetMergeEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspMergeEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==mpMap->GetInitKFid())
        {
            return;
        }
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
    {
        mit->first->EraseConnection(this);
    }

    for(size_t i=0; i<mvpMapPoints.size(); i++)
    {
        if(mvpMapPoints[i])
        {
            mvpMapPoints[i]->EraseObservation(this);
        }
    }

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        if(mpParent)
            sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty())
        {
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }
        }

        if(mpParent){
            mpParent->EraseChild(this);
            mTcp = mTcw * mpParent->GetPoseInverse();
        }
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}


vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r, const bool bRight, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (NLeft == -1) ? mvKeysUn[vCell[j]]
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

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

bool KeyFrame::UnprojectStereo(int i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        x3D = mRwc * x3Dc + mTwc.translation();
        return true;
    }
    else
        return false;
}

float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    if(N==0)
        return -1.0;

    vector<MapPoint*> vpMapPoints;
    Eigen::Matrix3f Rcw;
    Eigen::Vector3f tcw;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        tcw = mTcw.translation();
        Rcw = mRcw;
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    Eigen::Matrix<float,1,3> Rcw2 = Rcw.row(2);
    float zcw = tcw(2);
    for(int i=0; i<N; i++) {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            Eigen::Vector3f x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw) + zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

void KeyFrame::SetNewBias(const IMU::Bias &b)
{
    unique_lock<mutex> lock(mMutexPose);
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

Eigen::Vector3f KeyFrame::GetGyroBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
}

Eigen::Vector3f KeyFrame::GetAccBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bax, mImuBias.bay, mImuBias.baz);
}

IMU::Bias KeyFrame::GetImuBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return mImuBias;
}

Map* KeyFrame::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void KeyFrame::UpdateMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void KeyFrame::PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP, set<GeometricCamera*>& spCam)
{
    // Save the id of each MapPoint in this KF, there can be null pointer in the vector
    mvBackupMapPointsId.clear();
    mvBackupMapPointsId.reserve(N);
    for(int i = 0; i < N; ++i)
    {

        if(mvpMapPoints[i] && spMP.find(mvpMapPoints[i]) != spMP.end()) // Checks if the element is not null
            mvBackupMapPointsId.push_back(mvpMapPoints[i]->mnId);
        else // If the element is null his value is -1 because all the id are positives
            mvBackupMapPointsId.push_back(-1);
    }
    // Save the id of each connected KF with it weight
    mBackupConnectedKeyFrameIdWeights.clear();
    for(std::map<KeyFrame*,int>::const_iterator it = mConnectedKeyFrameWeights.begin(), end = mConnectedKeyFrameWeights.end(); it != end; ++it)
    {
        if(spKF.find(it->first) != spKF.end())
            mBackupConnectedKeyFrameIdWeights[it->first->mnId] = it->second;
    }

    // Save the parent id
    mBackupParentId = -1;
    if(mpParent && spKF.find(mpParent) != spKF.end())
        mBackupParentId = mpParent->mnId;

    // Save the id of the childrens KF
    mvBackupChildrensId.clear();
    mvBackupChildrensId.reserve(mspChildrens.size());
    for(KeyFrame* pKFi : mspChildrens)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupChildrensId.push_back(pKFi->mnId);
    }

    // Save the id of the loop edge KF
    mvBackupLoopEdgesId.clear();
    mvBackupLoopEdgesId.reserve(mspLoopEdges.size());
    for(KeyFrame* pKFi : mspLoopEdges)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupLoopEdgesId.push_back(pKFi->mnId);
    }

    // Save the id of the merge edge KF
    mvBackupMergeEdgesId.clear();
    mvBackupMergeEdgesId.reserve(mspMergeEdges.size());
    for(KeyFrame* pKFi : mspMergeEdges)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupMergeEdgesId.push_back(pKFi->mnId);
    }

    //Camera data
    mnBackupIdCamera = -1;
    if(mpCamera && spCam.find(mpCamera) != spCam.end())
        mnBackupIdCamera = mpCamera->GetId();

    mnBackupIdCamera2 = -1;
    if(mpCamera2 && spCam.find(mpCamera2) != spCam.end())
        mnBackupIdCamera2 = mpCamera2->GetId();

    //Inertial data
    mBackupPrevKFId = -1;
    if(mPrevKF && spKF.find(mPrevKF) != spKF.end())
        mBackupPrevKFId = mPrevKF->mnId;

    mBackupNextKFId = -1;
    if(mNextKF && spKF.find(mNextKF) != spKF.end())
        mBackupNextKFId = mNextKF->mnId;

    if(mpImuPreintegrated)
        mBackupImuPreintegrated.CopyFrom(mpImuPreintegrated);
}

void KeyFrame::PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid, map<unsigned int, GeometricCamera*>& mpCamId){
    // Rebuild the empty variables

    // Pose
    SetPose(mTcw);

    mTrl = mTlr.inverse();

    // Reference reconstruction
    // Each MapPoint sight from this KeyFrame
    mvpMapPoints.clear();
    mvpMapPoints.resize(N);
    for(int i=0; i<N; ++i)
    {
        if(mvBackupMapPointsId[i] != -1)
            mvpMapPoints[i] = mpMPid[mvBackupMapPointsId[i]];
        else
            mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
    }

    // Conected KeyFrames with him weight
    mConnectedKeyFrameWeights.clear();
    for(map<long unsigned int, int>::const_iterator it = mBackupConnectedKeyFrameIdWeights.begin(), end = mBackupConnectedKeyFrameIdWeights.end();
        it != end; ++it)
    {
        KeyFrame* pKFi = mpKFid[it->first];
        mConnectedKeyFrameWeights[pKFi] = it->second;
    }

    // Restore parent KeyFrame
    if(mBackupParentId>=0)
        mpParent = mpKFid[mBackupParentId];

    // KeyFrame childrens
    mspChildrens.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupChildrensId.begin(), end = mvBackupChildrensId.end(); it!=end; ++it)
    {
        mspChildrens.insert(mpKFid[*it]);
    }

    // Loop edge KeyFrame
    mspLoopEdges.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupLoopEdgesId.begin(), end = mvBackupLoopEdgesId.end(); it != end; ++it)
    {
        mspLoopEdges.insert(mpKFid[*it]);
    }

    // Merge edge KeyFrame
    mspMergeEdges.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupMergeEdgesId.begin(), end = mvBackupMergeEdgesId.end(); it != end; ++it)
    {
        mspMergeEdges.insert(mpKFid[*it]);
    }

    //Camera data
    if(mnBackupIdCamera >= 0)
    {
        mpCamera = mpCamId[mnBackupIdCamera];
    }
    else
    {
        cout << "ERROR: There is not a main camera in KF " << mnId << endl;
    }
    if(mnBackupIdCamera2 >= 0)
    {
        mpCamera2 = mpCamId[mnBackupIdCamera2];
    }

    //Inertial data
    if(mBackupPrevKFId != -1)
    {
        mPrevKF = mpKFid[mBackupPrevKFId];
    }
    if(mBackupNextKFId != -1)
    {
        mNextKF = mpKFid[mBackupNextKFId];
    }
    mpImuPreintegrated = &mBackupImuPreintegrated;


    // Remove all backup container
    mvBackupMapPointsId.clear();
    mBackupConnectedKeyFrameIdWeights.clear();
    mvBackupChildrensId.clear();
    mvBackupLoopEdgesId.clear();

    UpdateBestCovisibles();
}

bool KeyFrame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mRcw * P + mTcw.translation();
    float &PcX = Pc(0);
    float &PcY = Pc(1);
    float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    // cout << "c";

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

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

    float u_distort = x_distort * fx + cx;
    float v_distort = y_distort * fy + cy;

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

bool KeyFrame::ProjectPointUnDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mRcw * P + mTcw.translation();
    float &PcX = Pc(0);
    float &PcY= Pc(1);
    float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u = fx * PcX * invz + cx;
    v = fy * PcY * invz + cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    kp = cv::Point2f(u, v);

    return true;
}

Sophus::SE3f KeyFrame::GetRelativePoseTrl()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTrl;
}

Sophus::SE3f KeyFrame::GetRelativePoseTlr()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTlr;
}

Sophus::SE3<float> KeyFrame::GetRightPose() {
    unique_lock<mutex> lock(mMutexPose);

    return mTrl * mTcw;
}

Sophus::SE3<float> KeyFrame::GetRightPoseInverse() {
    unique_lock<mutex> lock(mMutexPose);

    return mTwc * mTlr;
}

Eigen::Vector3f KeyFrame::GetRightCameraCenter() {
    unique_lock<mutex> lock(mMutexPose);

    return (mTwc * mTlr).translation();
}

Eigen::Matrix<float,3,3> KeyFrame::GetRightRotation() {
    unique_lock<mutex> lock(mMutexPose);

    return (mTrl.so3() * mTcw.so3()).matrix();
}

Eigen::Vector3f KeyFrame::GetRightTranslation() {
    unique_lock<mutex> lock(mMutexPose);
    return (mTrl * mTcw).translation();
}

void KeyFrame::SetORBVocabulary(ORBVocabulary* pORBVoc)
{
    mpORBvocabulary = pORBVoc;
}

void KeyFrame::SetKeyFrameDatabase(KeyFrameDatabase* pKFDB)
{
    mpKeyFrameDB = pKFDB;
}

bool KeyFrame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
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

int KeyFrame::AddKeyPoint(int x, int y, int octave, float size, MapPoint* pMP)
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

} //namespace ORB_SLAM
