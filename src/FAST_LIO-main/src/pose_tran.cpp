#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h> 
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <ikd-Tree/ikd_Tree.h>
#include "ieskf_slam/math/geometry.h"
#include "ieskf_slam/math/SO3.h"
#include <mutex>
#include <fstream>
#include "std_msgs/Bool.h"
#include "std_msgs/String.h"
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <iostream>  
#include <vector>   
#include <thread>    
#include <chrono>  
#include <std_srvs/Trigger.h>
#include <opencv2/opencv.hpp>


// 自定义顶点基类，包含索引信息  
struct Vertex_info {  
    size_t index; // 指向原始点云中点的索引  
};  


typedef pcl::PointXYZ PointT;  



KD_TREE<PointType> ikdtree;
//初始参数
pcl::PointCloud<PointType>::Ptr init_cloud(new pcl::PointCloud<PointType>);



double x2, y2, z2, qx, qy, qz, qw; //初始位姿变化
double startx,starty,startz,startqx,startqy,startqz,startqw;
Eigen::Quaterniond rotation,strat_rotation;
Eigen::Vector3d position,strat_position;
int iter_times;
bool init_flag = false;
bool converge =true;

//转化后的地图点云
pcl::PointCloud<PointType>::Ptr tran_initcloud(new pcl::PointCloud<PointType>);

//可视化特征点云
pcl::PointCloud<pcl::PointXYZRGB>::Ptr effect_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);


//定义协方差矩阵
Eigen::Matrix<double,18,18> P_init;

//收集的帧数
int frame_num = 0;

//icp匹配部分参数
template<typename _first, typename _second, typename _thrid>
struct triple{
        _first first;
        _second second;
        _thrid thrid;
};
using loss_type = triple<Eigen::Vector3d,Eigen::Vector3d,double>; //残差定义

//定义状态
struct State18
{
        Eigen::Quaterniond rotation;
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        Eigen::Vector3d bg;
        Eigen::Vector3d ba;
        Eigen::Vector3d gravity;
        State18(){
                rotation = Eigen::Quaterniond::Identity();
                position = Eigen::Vector3d::Zero();
                velocity = Eigen::Vector3d::Zero();
                bg = Eigen::Vector3d::Zero();
                ba = Eigen::Vector3d::Zero();
                gravity = Eigen::Vector3d::Zero();
        }
};

//定义初始状态
State18 X_init;
State18 x_k_k ;

Eigen::Matrix<double,18,1> getErrorState18(const State18 &s1, const  State18 &s2){
        Eigen::Matrix<double,18,1> es;
        es.setZero();
        es.block<3,1>(0,0) = SO3Log(s2.rotation.toRotationMatrix().transpose() * s1.rotation.toRotationMatrix());
        es.block<3,1>(3,0) = s1.position - s2.position;
        es.block<3,1>(6,0) = s1.velocity - s2.velocity;
        es.block<3,1>(9,0) = s1.bg - s2.bg;
        es.block<3,1>(12,0) = s1.ba - s2.ba;
        es.block<3,1>(15,0) = s1.gravity - s2.gravity;
        return es;
}


std::mutex laser_mtx;
std::mutex odom_mtx;

std::queue<sensor_msgs::PointCloud2::ConstPtr> laser_buffer;
std::queue<nav_msgs::Odometry::ConstPtr> odom_buffer;

typedef pcl::PointCloud<PointType> PointCloud;



double cur_z =0;
void OdomHandler(const nav_msgs::Odometry::ConstPtr &msg) {
  std::unique_lock<std::mutex> lock(odom_mtx);
  cur_z = msg->pose.pose.position.z;
  odom_buffer.push(msg);
}




///////迭代计算部分代码
bool calculate(const State18 &state,pcl::PointCloud<PointType>::Ptr &cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr &effect_cloud, Eigen::MatrixXd & Z,Eigen::MatrixXd & H)
{
        std::vector<loss_type> loss_v;
        loss_v.resize(cloud->size());
        std::vector<bool> is_effect_point(cloud->size(),false);
        std::vector<loss_type> loss_real;
        int  vaild_points_num = 0;
        /**
         * 有效点的判断
         * 1. 将当前点变换到世界系下
         * 2. 在局部地图上使用kd_tree 临近搜索NEAR_POINTS_NUM个点
         * 3. 判断这些点是否构成平面
         * 4. 判断点离这个平面够不够近(达到阈值)
         * 5. 满足上述条件，设置为有效点。
         */
        for (size_t  i = 0; i < cloud->size(); i++)
        {
                // . 变换到世界系
                PointType point_init = cloud->points[i];
                PointType point_pcd , point_world;
                point_pcd = transformPoint(point_init,strat_rotation,strat_position);
                point_world = transformPoint(point_pcd,state.rotation,state.position);
                // . 临近搜索;
                // tran_initcloud->push_back(point_pcd);
                const int NEAR_POINTS_NUM = 5;
                PointVector point_ind;
                vector<float> distance(NEAR_POINTS_NUM); ////==这个地方不写括号，出现段错误！！
                
                //printf("66666--------------------------------\n");
                ikdtree.Nearest_Search(point_world,NEAR_POINTS_NUM,point_ind,distance);

                if (distance.size()<NEAR_POINTS_NUM||distance[NEAR_POINTS_NUM-1]>6)
                {
                        continue;
                }
                // . 判断这些点够不够成平面
                std::vector<PointType> planar_points;
                for (int ni = 0; ni < NEAR_POINTS_NUM; ni++)
                {
                        planar_points.push_back(point_ind[ni]);
                }
                Eigen::Vector4d pabcd;
                // . 如果构成平面

                if (planarCheck(planar_points,pabcd,0.1))    
                {
                        // . 计算点到平面距离
                        double pd = point_world.x*pabcd(0)+point_world.y*pabcd(1)+point_world.z*pabcd(2)+pabcd(3);
                        // . 记录残差
                        loss_type loss;
                        loss.thrid = pd; // 残差
                        loss.first = {point_pcd.x,point_pcd.y,point_pcd.z}; // imu系下点的坐标，用于求H
                        loss.second = pabcd.block<3,1>(0,0);// 平面法向量 用于求H
                        if (isnan(pd)||isnan(loss.second(0))||isnan(loss.second(1))||isnan(loss.second(2)))
                        {
                                printf("isnan!!\n");
                                continue;
                        }
                        // .计算点和平面的夹角，夹角越小S越大。
                        double s = 1 - 0.9 * fabs(pd) / sqrt(loss.first.norm());
                        if(s > 0.9 )
                        {
                                vaild_points_num++;
                                loss_v[i]=(loss);
                                is_effect_point[i] = true;
                        }
                }


        }
        for (size_t i = 0; i <cloud->size(); i++)
        {
                if(is_effect_point[i])
                {
                        loss_real.push_back(loss_v[i]);
                        
                        pcl::PointXYZRGB point;
                        point.x = cloud->points[i].x;
                        point.y = cloud->points[i].y;
                        point.z = cloud->points[i].z;
                        point.r = 255;
                        point.g = 0;
                        point.b = 0;
                        effect_cloud->push_back(point);

                }
        
        }
        // 根据有效点的数量分配H Z的大小
        vaild_points_num = loss_real.size();
        H = Eigen::MatrixXd::Zero(vaild_points_num, 18); 
        Z.resize(vaild_points_num,1);
        for (int vi = 0; vi < vaild_points_num; vi++)
        {
                // H 记录导数
                Eigen::Vector3d dr = -1*loss_real[vi].second.transpose()*state.rotation.toRotationMatrix()*skewSymmetric(loss_real[vi].first);
                H.block<1,3>(vi,0) = dr.transpose();
                H.block<1,3>(vi,3) = loss_real[vi].second.transpose();
                // Z记录距离
                Z(vi,0) = loss_real[vi].thrid;
        }
        return true;
}

bool update(pcl::PointCloud<PointType>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &effect_cloud)
{
        x_k_k = X_init;
        ///迭代
        Eigen::MatrixXd K;
        Eigen::MatrixXd H_k;
        Eigen::Matrix<double,18,18> P_in_update;
        int n =0;
        for (int i = 0; i < iter_times; i++)
        {
                ///. 计算误差状态 J 
                Eigen::Matrix<double,18,1> error_state = getErrorState18(x_k_k,X_init);
                //printf("x_kk first: %f, second: %f\n",x_k_k.position.x(),x_k_k.position.y());
                Eigen::Matrix<double,18,18> J_inv;
                J_inv.setIdentity();            
                J_inv.block<3,3>(0,0) = A_T(error_state.block<3,1>(0,0));
                // 更新 P
                P_in_update = J_inv*P_init*J_inv.transpose();

                Eigen::MatrixXd z_k;
                //printf("444444444--------------------------------\n");
                // 调用接口计算 Z H
                calculate(x_k_k,cloud,effect_cloud,z_k,H_k);
                // printf("777777--------------------------------\n");
                Eigen::MatrixXd H_kt = H_k.transpose();
                // R 直接写死0.001; 
                K = (H_kt*H_k+(P_in_update/0.001).inverse()).inverse()*H_kt;
                //. 计算X 的增量
                Eigen::MatrixXd left = -1*K*z_k;
                Eigen::MatrixXd right = -1*(Eigen::Matrix<double,18,18>::Identity()-K*H_k)*J_inv*error_state; 
                Eigen::MatrixXd update_x = left+right;

                // 收敛判断
                n++;
                converge =true;
                // for ( int idx = 0; idx < 6; idx++)
                // {
                //         printf("第 %d 次的 update x: %f\n", n,update_x(idx,0 )); 
                //         if (update_x(idx,0)>0.001)
                //         {
      
                //                 converge = false;
                //                 break;
                //         }
                
                // }
                double tol = 1e-3;
                if (update_x.block<6,1>(0,0).cwiseAbs().maxCoeff() > tol) {
                converge = false;
                } else {
                converge = true;
                }


                // 更新X
                x_k_k.rotation = x_k_k.rotation.toRotationMatrix()*so3Exp(update_x.block<3,1>(0,0));
                x_k_k.rotation.normalize();
                x_k_k.position = x_k_k.position+update_x.block<3,1>(3,0);
                x_k_k.velocity = x_k_k.velocity+update_x.block<3,1>(6,0);
                x_k_k.bg = x_k_k.bg+update_x.block<3,1>(9,0);
                x_k_k.ba = x_k_k.ba+update_x.block<3,1>(12,0);
                x_k_k.gravity = x_k_k.gravity+update_x.block<3,1>(15,0);
                if(converge){
                        break;
                }else{
                        effect_cloud->clear();
                }
        }
        printf("iterations count is : %d\n",n);
        X_init = x_k_k;
        printf("================================\n");
        printf("T: %f ,%f ,%f \nR: %f , %f, %f, %f\n",X_init.position.x() ,X_init.position.y(),X_init.position.z(),X_init.rotation.w(),X_init.rotation.x(),X_init.rotation.y(),X_init.rotation.z());
        
        // std::ofstream outfile("output.txt");
        // if (!outfile.is_open()) {
        //         std::cerr << "Failed to open file for writing." << std::endl;
        //         return 1;
        // }
        // // 写入值到文件
        // outfile << "T: " << X_init.position.x() << "," << X_init.position.y() << "," << X_init.position.z() << "\n"
        //         << "R: " << X_init.rotation.w() << ", " << X_init.rotation.x() << ", " << X_init.rotation.y() << ", " << X_init.rotation.z() << std::endl;
        // // 关闭文件
        // outfile.close();

        P_init = (Eigen::Matrix<double,18,18>::Identity()-K*H_k)*P_in_update;
        return converge;
}




/////点云处理////
pcl::PointCloud<PointType>::Ptr accumulatedCloud(new pcl::PointCloud<PointType>());

// 回调函数，处理接收到的点云数据
void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& inputCloud) {
    // 累积点云数据
    if(!init_flag)
    {
        // 将ROS的PointCloud2消息转换为PCL的PointCloud
        pcl::PointCloud<PointType>::Ptr tempCloud(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(*inputCloud, *tempCloud);
        *accumulatedCloud += *tempCloud;
    }
    
}


void publish_tran(const ros::Publisher & posetran)
{
        nav_msgs::Odometry new_odom1 ;
        Eigen::Quaterniond q_start(startqw, startqx, startqy, startqz);
        Eigen::Quaterniond combined = q_start * X_init.rotation; // or swap order per convention
        Eigen::Vector3d combined_pos = q_start * X_init.position + Eigen::Vector3d(startx,starty,startz);
        
        new_odom1.pose.pose.position.x = combined_pos.x();
        new_odom1.pose.pose.position.y = combined_pos.y();
        new_odom1.pose.pose.position.z = combined_pos.z();
   

        new_odom1.pose.pose.orientation.w = combined.w();
        new_odom1.pose.pose.orientation.x = combined.x();
        new_odom1.pose.pose.orientation.y = combined.y();
        new_odom1.pose.pose.orientation.z = combined.z();  
        posetran.publish(new_odom1);
}


/////////////////////
int main(int argc, char** argv)
{
        ros::init(argc, argv,"pose_transform");
        ros::NodeHandle nh;

        ros::Subscriber subOdom = nh.subscribe<nav_msgs::Odometry>("/b/Odometry", 100, OdomHandler);
        ros::Subscriber sub_cloud = nh.subscribe("/b/cloud_registered", 1000, cloudCallback);
        
        ros::Publisher posetran =nh.advertise<nav_msgs::Odometry>("/b/output_pose", 10);


        //读取参数
        string icpmap_path;
        nh.param<string>("pose_transform/icpmap_path",icpmap_path, " ");


        nh.param<double>("pose_transform/startx", startx, 0);
        nh.param<double>("pose_transform/starty", starty, 0);
        nh.param<double>("pose_transform/startz", startz, 0);
        nh.param<double>("pose_transform/startqx", startqx, 0);
        nh.param<double>("pose_transform/startqy", startqy, 0);
        nh.param<double>("pose_transform/startqz", startqz, 0);
        nh.param<double>("pose_transform/startqw", startqw, 1);

        double z_thresholds;
        nh.param<double>("pose_transform/z_thresholds", z_thresholds, 0);
        nh.param<int>("pose_transform/iter_times", iter_times, 10);

        strat_rotation = Eigen::Quaterniond(startqw, startqx, startqy, startqz);
        strat_position << startx, starty, startz;


        P_init.setIdentity();
        P_init(9,9)   = P_init(10,10) = P_init(11,11) = 0.0001;
        P_init(12,12) = P_init(13,13) = P_init(14,14) = 0.001;
        P_init(15,15) = P_init(16,16) = P_init(17,17) = 0.00001; 


        //读取pcd文件
        pcl::PointCloud<PointType>::Ptr pcd_cloud(new pcl::PointCloud<PointType>);
        if (pcl::io::loadPCDFile<PointType>(icpmap_path, *pcd_cloud) == -1) 
        {
                PCL_ERROR("Couldn't read file111\n");
                return (-1);
        }

        //将点云加入到ikdtree中
        if(ikdtree.Root_Node == nullptr)
        {
                ikdtree.set_downsample_param(0.2);
                ikdtree.Build(pcd_cloud->points);
        }
        
        ros::Rate wait(10);
        //累积点云
        // while (cur_z < z_thresholds && ros::ok())
        // {
        //         ros::spinOnce();
        //         wait.sleep();
        // }

        //按照时间
        ros::Time start_time = ros::Time::now();
        ros::Duration timeout(10.0); // 3秒超时

        while ((ros::Time::now() - start_time < timeout))
        {
                ros::spinOnce();
                wait.sleep();
        }

        pcl::PointCloud<PointType>::Ptr filteredCloud(new pcl::PointCloud<PointType>);
        //进行体素滤波
        pcl::VoxelGrid<PointType> voxelGrid;
        voxelGrid.setInputCloud(accumulatedCloud);
        voxelGrid.setLeafSize(0.3, 0.3, 0.3); // 设置体素大小
        voxelGrid.filter(*filteredCloud);


        //进行icp匹配，挑选有效点

        if (update(filteredCloud ,effect_cloud))
        {

                ROS_INFO("icp is successful!!   starting put out new odometry!!");
                init_flag = true;
                ros::Rate rate(100);

                while(ros::ok())
                {
                        ros::spinOnce();
                        publish_tran(posetran);
                        rate.sleep();
        
                }
        
        }else{
                ROS_WARN("icp is not converge!!");

        }
                
        
        pcl::io::savePCDFileASCII(string(ROOT_DIR)+"PCD/map.pcd", *filteredCloud);
        printf("save map!! \n");
        pcl::io::savePCDFileASCII(string(ROOT_DIR)+"PCD/tran.pcd", *tran_initcloud);
        printf("save tran cloud!! \n");
        pcl::io::savePCDFileASCII(string(ROOT_DIR)+"PCD/effect_cloud.pcd", *effect_cloud);
        printf("save effect cloud!! \n");

        return 0;

}