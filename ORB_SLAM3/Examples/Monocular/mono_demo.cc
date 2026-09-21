/**
* Based on mono_euroc.cc
*
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<MatchSaver.h>
#include<OpticalFlowLoader.h>
#include<System.h>


using namespace std;

ORB_SLAM3::MatchSaver* g_pMatchSaver = nullptr;
ORB_SLAM3::OpticalFlowLoader* g_pOpticalFlowLoader = nullptr;

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps, vector<int64_t> &vTimeStampsNanoseconds);

int main(int argc, char **argv)
{  
    if(argc != 8)
    {
        cerr << endl << "Usage: ./mono_tartanair path_to_vocabulary path_to_settings path_to_image_folder path_to_times_file output_traj_file_name path_to_opticalflow_file first_frame_id" << endl;
        return 1;
    }

    // Load all sequences:
    int seq;
    vector<string>vstrImageFilenames;
    vector<double>vTimestampsCam;
    vector<int64_t>vTimestampsCamNanoseconds;
    int nImages;

    cout << "Loading images for sequence " << seq << "...";
    LoadImages(string(argv[3]), string(argv[4]), vstrImageFilenames, vTimestampsCam, vTimestampsCamNanoseconds);
    nImages = vstrImageFilenames.size();
    cout << "LOADED!" << endl;

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Load optical flow
    std::string opticalFlowPath = argv[6];
    g_pOpticalFlowLoader = new ORB_SLAM3::OpticalFlowLoader(opticalFlowPath);
    if (!g_pOpticalFlowLoader->IsLoaded())
    {
        std::cerr << "Failed to load optical flow data from: " << opticalFlowPath << std::endl;
    }

    // first frame index
    int initFr = argv[7] ? std::stoi(argv[7]) : 0;
    cout << "First Frame Index: " << initFr << endl;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    cv::Size imSize = SLAM.GetImageSize();

    // Class to save feature matches
    // comment out to avoid save matches
    // g_pMatchSaver = new ORB_SLAM3::MatchSaver("matches.bin", imSize.width, imSize.height);

    double t_resize = 0.f;
    double t_track = 0.f;

    // Core 
    cv::Mat im;
    for(int ni=initFr; ni<nImages-5; ni++)
    {
        if (ni % 100 == 0)
        {
            cout << "===========================================================  Processing frame " << ni << " of " << nImages << endl;
        }

        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],cv::IMREAD_UNCHANGED); //,CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestampsCam[ni];
        int64_t tframeNs = vTimestampsCamNanoseconds[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                    <<  vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
#endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
#endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        // cout << "tframe = " << tframe << endl;
        SLAM.TrackMonocular(im, tframe, tframeNs); // TODO change to monocular_inertial

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestampsCam[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestampsCam[ni-1];

        if(ttrack<T) {
            //std::cout << "usleep: " << (dT-ttrack) << std::endl;
            usleep((T-ttrack)*1e6);
        }
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    std::string trajectory_file_name = argv[5];
    SLAM.SaveAllMapTrajectoriesEuRoC(trajectory_file_name);
    // SLAM.SaveKeyFrameTrajectoryEuRoC(trajectory_file_name + "_keyframes.txt");

    // Clean up optical flow loader
    if (g_pOpticalFlowLoader)
    {
        delete g_pOpticalFlowLoader;
        g_pOpticalFlowLoader = nullptr;
    }

    return 0;
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps, vector<int64_t> &vTimeStampsNanoseconds)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            int64_t t;
            ss >> t;
            vTimeStamps.push_back(static_cast<double>(t)*1e-9);
            vTimeStampsNanoseconds.push_back(t);

        }
    }
}
