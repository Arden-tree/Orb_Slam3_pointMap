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
#include <unordered_map>

namespace ORB_SLAM3
{

// Simple grid-hash voxel filter (replaces PCL VoxelGrid to avoid heap corruption)
static PointCloud::Ptr voxelFilter(const PointCloud::Ptr& cloud, float leaf)
{
    PointCloud::Ptr out(new PointCloud());
    if (!cloud || cloud->empty())
        return out;

    struct CellKey {
        int cx, cy, cz;
        bool operator==(const CellKey& o) const { return cx==o.cx && cy==o.cy && cz==o.cz; }
    };
    struct CellHash {
        size_t operator()(const CellKey& k) const {
            return ((size_t)k.cx * 73856093) ^ ((size_t)k.cy * 19349669) ^ ((size_t)k.cz * 83492791);
        }
    };

    float inv = 1.0f / leaf;
    std::unordered_map<CellKey, PointT, CellHash> grid;
    grid.reserve(cloud->size() / 2);

    for (const auto& p : cloud->points)
    {
        CellKey key{(int)std::floor(p.x * inv), (int)std::floor(p.y * inv), (int)std::floor(p.z * inv)};
        auto it = grid.find(key);
        if (it == grid.end())
        {
            grid.insert({key, p});
        }
        else
        {
            // Average color
            auto& avg = it->second;
            avg.x = (avg.x + p.x) * 0.5f;
            avg.y = (avg.y + p.y) * 0.5f;
            avg.z = (avg.z + p.z) * 0.5f;
            avg.r = (avg.r + p.r) / 2;
            avg.g = (avg.g + p.g) / 2;
            avg.b = (avg.b + p.b) / 2;
        }
    }

    out->points.reserve(grid.size());
    for (const auto& kv : grid)
        out->points.push_back(kv.second);

    out->width = out->points.size();
    out->height = 1;
    out->is_dense = false;
    return out;
}

PointCloudMapping::PointCloudMapping(double resolution)
    : mResolution(resolution), mVoxelLeafSize(resolution * 2.0), mbShutDown(false)
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
    return PointCloud::Ptr(new PointCloud(*mGlobalMap));
}

PointCloud::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    PointCloud::Ptr cloud(new PointCloud());

    Eigen::Isometry3d T = toIsometry3d(kf->GetPoseInverse());
    Eigen::Matrix3d R = T.rotation();
    Eigen::Vector3d t = T.translation();

    for (int m = 0; m < depth.rows; m += 4)
    {
        for (int n = 0; n < depth.cols; n += 4)
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

            // Global voxel filter to cap total size
            if (mGlobalMap->size() > 200000)
            {
                mGlobalMap = voxelFilter(mGlobalMap, mVoxelLeafSize);
                std::cout << "[PointCloud] global voxel: " << mGlobalMap->size() << " points" << std::endl;
            }
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
