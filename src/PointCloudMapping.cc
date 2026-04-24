/**
* This file is part of ORB-SLAM3
*
* Point Cloud Mapping - dense point cloud generation, rendered by MapDrawer
*/

#include "PointCloudMapping.h"
#include "KeyFrame.h"
#include "Converter.h"

#include <iostream>
#include <cmath>

namespace ORB_SLAM3
{

PointCloudMapping::PointCloudMapping(double resolution)
    : mResolution(resolution), mbShutDown(false)
{
    mGlobalMap.reset(new PointCloud());
    mThread = std::make_shared<std::thread>(&PointCloudMapping::run, this);
}

PointCloudMapping::~PointCloudMapping()
{
}

void PointCloudMapping::shutdown()
{
    {
        std::unique_lock<std::mutex> lck(mQueueMutex);
        mbShutDown = true;
    }
    mQueueCV.notify_one();
    if (mThread && mThread->joinable())
        mThread->join();

    // Save
    std::unique_lock<std::mutex> lck(mMapMutex);
    if (mGlobalMap && !mGlobalMap->empty())
    {
        pcl::io::savePCDFileBinary("PointCloudOutput.pcd", *mGlobalMap);
        std::cout << "[PointCloud] saved " << mGlobalMap->size() << " points to PointCloudOutput.pcd" << std::endl;
    }
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    std::cout << "[PointCloud] receive a keyframe, id = " << kf->mnId << std::endl;
    {
        std::unique_lock<std::mutex> lck(mQueueMutex);
        mKeyframeQueue.push_back({kf, {color.clone(), depth.clone()}});
    }
    mQueueCV.notify_one();
}

PointCloud::Ptr PointCloudMapping::getGlobalMap()
{
    std::unique_lock<std::mutex> lck(mMapMutex);
    return mGlobalMap;
}

PointCloud::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    PointCloud::Ptr cloud(new PointCloud());

    Eigen::Isometry3d T = toIsometry3d(kf->GetPoseInverse());
    Eigen::Matrix3d R = T.rotation();
    Eigen::Vector3d t = T.translation();

    for (int m = 0; m < depth.rows; m += 2)
    {
        for (int n = 0; n < depth.cols; n += 2)
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01f || d > 10.0f)
                continue;

            float x = (n - kf->cx) * d / kf->fx;
            float y = (m - kf->cy) * d / kf->fy;
            float z = d;

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                continue;

            PointT p;
            p.x = static_cast<float>(R(0,0)*x + R(0,1)*y + R(0,2)*z + t(0));
            p.y = static_cast<float>(R(1,0)*x + R(1,1)*y + R(1,2)*z + t(1));
            p.z = static_cast<float>(R(2,0)*x + R(2,1)*y + R(2,2)*z + t(2));

            if (color.type() == CV_8UC3)
            {
                p.b = color.ptr<uchar>(m)[n * 3];
                p.g = color.ptr<uchar>(m)[n * 3 + 1];
                p.r = color.ptr<uchar>(m)[n * 3 + 2];
            }
            else
            {
                uchar gray = color.ptr<uchar>(m)[n];
                p.b = gray; p.g = gray; p.r = gray;
            }

            cloud->points.push_back(p);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    std::cout << "[PointCloud] kf " << kf->mnId << " -> " << cloud->size() << " points" << std::endl;
    return cloud;
}

void PointCloudMapping::run()
{
    while (true)
    {
        KeyFrame* kf;
        cv::Mat color, depth;
        {
            std::unique_lock<std::mutex> lck(mQueueMutex);
            mQueueCV.wait(lck, [this] { return !mKeyframeQueue.empty() || mbShutDown; });

            if (mbShutDown && mKeyframeQueue.empty())
                break;

            if (mKeyframeQueue.empty())
                continue;

            auto entry = mKeyframeQueue.front();
            kf = entry.first;
            color = entry.second.first;
            depth = entry.second.second;
            mKeyframeQueue.erase(mKeyframeQueue.begin());
        }

        PointCloud::Ptr cloud = generatePointCloud(kf, color, depth);
        if (!cloud->empty())
        {
            std::unique_lock<std::mutex> lck(mMapMutex);
            *mGlobalMap += *cloud;
        }
    }
}

Eigen::Isometry3d PointCloudMapping::toIsometry3d(const Sophus::SE3f& pose)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    Eigen::Matrix3f Rf = pose.rotationMatrix();
    Eigen::Vector3f tf = pose.translation();
    Eigen::Matrix3d Rd = Rf.cast<double>();
    Eigen::Vector3d td = tf.cast<double>();
    T.linear() = Rd;
    T.translation() = td;
    return T;
}

} // namespace ORB_SLAM3
