#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>

class Snapshot{

    public:
    Snapshot(){
        visualization_msgs::MarkerArray octomap_scan;
        sub_markers=nh.subscribe("/octomap",5,&Snapshot::cb_octomap,this);
        sub_takeshot=nh.subscribe("/snapshot",5,&Snapshot::cb_snapshot,this)
    }
    void cb_snapshot(){


    }
    void cb_octomap(visualization_msgs::MarkerArray octomap_scan){
        this.octomap_scan=octomap_scan;

    }
}