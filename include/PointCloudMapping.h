/**
* This file is part of ORB-SLAM3
*
* Point Cloud Mapping module - dense point cloud from RGB-D keyframes
* Rendered via Pangolin in MapDrawer, PCL used only for data + saving
*/

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include <condition_variable>
#include <thread>
#include <memory>
#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include "sophus/se3.hpp"

using PointT = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<PointT>;

namespace ORB_SLAM3
{

class KeyFrame;

class PointCloudMapping
{
public:
    PointCloudMapping(double resolution);
    ~PointCloudMapping();

    void insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    void shutdown();

    // Thread-safe access for MapDrawer to render
    PointCloud::Ptr getGlobalMap();

private:
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    void run(); // background processing loop

    double mResolution;
    double mVoxelLeafSize; // voxel filter leaf size for rendering
    PointCloud::Ptr mGlobalMap;
    std::mutex mMapMutex;

    std::shared_ptr<std::thread> mThread;

    std::vector<std::pair<KeyFrame*, std::pair<cv::Mat, cv::Mat>>> mKeyframeQueue;
    std::mutex mQueueMutex;
    std::condition_variable mQueueCV;
    bool mbShutDown;

    static Eigen::Isometry3d toIsometry3d(const Sophus::SE3f& pose);
};

} // namespace ORB_SLAM3

#endif // POINTCLOUDMAPPING_H
