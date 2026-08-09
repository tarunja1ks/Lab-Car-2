#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>
#include <octomap/octomap.h>
#include <std_msgs/String.h>
#include <octomap_msgs/conversions.h>

class Snapshot{
    ros::Subscriber sub_markers;
    ros::Subscriber sub_takeshot;
    octomap::OcTree* octomap_scan;

    public:
    Snapshot(ros::NodeHandle& nh):octomap_scan(nullptr){
        sub_markers=nh.subscribe("/octomap",5,&Snapshot::cb_octomap,this);
        sub_takeshot=nh.subscribe("/snapshot",5,&Snapshot::cb_snapshot,this);
    }
    ~Snapshot(){ delete octomap_scan; }

    void cb_snapshot(const std_msgs::String::ConstPtr &msg){
        if(!octomap_scan) return;
        octomap_scan->writeBinary("/Users/tarunjaikumar/Documents/Robotics-UCSD/ERL-Racecar/catkin_ws/src/mushr_gazebo/exports/octree.bt");
    }
    void cb_octomap(const octomap_msgs::Octomap::ConstPtr &msg){
        octomap::AbstractOcTree* t = octomap_msgs::binaryMsgToMap(*msg);
        if(!t) return;
        octomap::OcTree* nt = dynamic_cast<octomap::OcTree*>(t);
        if(!nt){ delete t; return; }
        delete octomap_scan;
        octomap_scan = nt;
    }
};

int main(int argc, char** argv){
  ros::init(argc, argv, "snapshot_node");
  ros::NodeHandle nh;
  Snapshot node(nh);
  ros::spin();
}