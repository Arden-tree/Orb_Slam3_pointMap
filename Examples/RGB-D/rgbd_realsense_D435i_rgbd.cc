/**
* This file is part of ORB-SLAM3
*
* RealSense D435i RGB-D only (no IMU) with split-view dense point cloud viewer.
* Based on rgbd_realsense_D435i.cc but simplified for pure RGB-D mode.
*/

#include <signal.h>
#include <iostream>
#include <chrono>
#include <thread>

#include <opencv2/core/core.hpp>
#include <librealsense2/rs.hpp>

#include <System.h>

using namespace std;

bool b_continue_session = true;

void exit_loop_handler(int s) {
    cout << "Finishing session" << endl;
    b_continue_session = false;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        cerr << endl
             << "Usage: ./rgbd_realsense_D435i_rgbd path_to_vocabulary path_to_settings (trajectory_file_name)"
             << endl;
        return 1;
    }

    string file_name;
    if (argc == 4) {
        file_name = string(argv[argc - 1]);
    }

    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    // --- RealSense setup with retry ---
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    rs2::pipeline_profile profile;
    float depth_scale = 1.0f;
    rs2_intrinsics intrinsics{};

    const int max_retries = 5;
    bool device_ready = false;
    for (int i = 0; i < max_retries; i++) {
        try {
            cout << "[D435i] Attempt " << (i + 1) << "/" << max_retries << " to start device..." << endl;
            profile = pipe.start(cfg);

            auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
            depth_scale = depth_sensor.get_depth_scale();

            rs2::video_stream_profile color_profile =
                profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
            intrinsics = color_profile.get_intrinsics();

            device_ready = true;
            break;
        } catch (const rs2::error &e) {
            cerr << "[D435i] Start failed: " << e.what() << endl;
            if (i < max_retries - 1) {
                cout << "[D435i] Retrying in 2 seconds..." << endl;
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
    }

    if (!device_ready) {
        cerr << "[D435i] Failed to start after " << max_retries << " attempts. Is the device connected?" << endl;
        return 1;
    }

    cout << "[D435i] depth scale = " << depth_scale << endl;
    cout << "[D435i] fx=" << intrinsics.fx << " fy=" << intrinsics.fy
         << " cx=" << intrinsics.ppx << " cy=" << intrinsics.ppy
         << " (" << intrinsics.width << "x" << intrinsics.height << ")" << endl;

    // Align depth to color
    rs2::align align(RS2_STREAM_COLOR);

    // Create SLAM system (RGBD mode, viewer enabled)
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, true, 0, file_name);
    float imageScale = SLAM.GetImageScale();

    cout << endl << "-------" << endl;
    cout << "D435i RGB-D live session started. Ctrl+C to stop." << endl;

    try {
        while (!SLAM.isShutDown() && b_continue_session) {
            rs2::frameset frames = pipe.wait_for_frames();
            rs2::frameset aligned = align.process(frames);

            rs2::video_frame color_frame = aligned.get_color_frame();
            rs2::depth_frame depth_frame = aligned.get_depth_frame();

            if (!color_frame || !depth_frame)
                continue;

            cv::Mat im(cv::Size(640, 480), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat depth(cv::Size(640, 480), CV_16U, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);

            // Convert depth to float (meters) - RealSense D435i depth_scale is typically 0.001
            cv::Mat depth_f;
            depth.convertTo(depth_f, CV_32F, depth_scale);

            if (imageScale != 1.f) {
                int width = im.cols * imageScale;
                int height = im.rows * imageScale;
                cv::resize(im, im, cv::Size(width, height));
                cv::resize(depth_f, depth_f, cv::Size(width, height));
            }

            double timestamp = std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count();

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif

            SLAM.TrackRGBD(im, depth_f, timestamp);

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
    #endif

            double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
            cout << "\rTracking time: " << ttrack * 1000 << " ms" << flush;
        }
    } catch (const rs2::error &e) {
        cerr << endl << "[D435i] RealSense error: " << e.what() << endl;
    } catch (const std::exception &e) {
        cerr << endl << "[D435i] Error: " << e.what() << endl;
    }

    cout << endl << "Shutting down..." << endl;

    try { pipe.stop(); } catch (...) {}

    SLAM.Shutdown();

    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    cout << "Trajectories saved." << endl;
    return 0;
}
