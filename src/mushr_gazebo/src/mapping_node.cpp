// mapping_node.cpp
// Subscribes to /cloud (PointCloud2 in the laser frame), transforms each scan
// into the map frame via tf, inserts it into an octree, and publishes the
// occupied voxels as a visualization_msgs/MarkerArray on /octomap_markers,
// rendered with rviz's built-in MarkerArray display.

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <map>
#include <vector>

class Mapper {
  ros::Subscriber sub_;
  ros::Publisher pub_rviz;
  ros::Publisher pub_octree;
  tf::TransformListener tf_;
  octomap::OcTree tree_;
  std::string map_frame_;
  double max_range_;
  double extrude_min_;
  double extrude_max_;

public:
  Mapper(ros::NodeHandle& nh) : tree_(0.05), map_frame_("map"), max_range_(10.0),
                                extrude_min_(0.0), extrude_max_(0.5) {
    ros::NodeHandle pnh("~");
    pnh.param("max_range", max_range_, max_range_);
    pnh.param("extrude_min", extrude_min_, extrude_min_);
    pnh.param("extrude_max", extrude_max_, extrude_max_);
    double res = tree_.getResolution();
    pnh.param("resolution", res, res);
    tree_.setResolution(res);
    ROS_INFO("mapping_node: resolution=%.3f max_range=%.1f extrude=[%.2f,%.2f]",
             res, max_range_, extrude_min_, extrude_max_);
    sub_ = nh.subscribe("/cloud", 5, &Mapper::cb, this);
    pub_rviz = nh.advertise<visualization_msgs::MarkerArray>("/octomap_markers", 1, true);
    pub_octree = nh.advertise<octomap_msgs::Octomap>("/octomap", 1, true);
  }

  void cb(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    tf::StampedTransform T;
    try {
      tf_.waitForTransform(map_frame_, msg->header.frame_id,
                           msg->header.stamp, ros::Duration(0.1));
      tf_.lookupTransform(map_frame_, msg->header.frame_id,
                          msg->header.stamp, T);
    } catch (tf::TransformException& e) {
      ROS_WARN_THROTTLE(1.0, "tf lookup failed: %s", e.what());
      return;
    }

    octomap::point3d origin(T.getOrigin().x(), T.getOrigin().y(), T.getOrigin().z());

    octomap::Pointcloud pc;
    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x"), iy(*msg, "y"), iz(*msg, "z");
    for (; ix != ix.end(); ++ix, ++iy, ++iz) {
      if (!std::isfinite(*ix)) continue;
      tf::Vector3 pm = T * tf::Vector3(*ix, *iy, *iz);
      pc.push_back(pm.x(), pm.y(), pm.z());
    }

    const double res = tree_.getResolution();
    if (extrude_max_ > extrude_min_) {
      for (double z = extrude_min_; z <= extrude_max_ + 1e-9; z += res) {
        octomap::Pointcloud layer;
        layer.reserve(pc.size());
        for (size_t i = 0; i < pc.size(); ++i)
          layer.push_back(pc[i].x(), pc[i].y(), z);
        tree_.insertPointCloud(layer, octomap::point3d(origin.x(), origin.y(), z),
                               max_range_, true, true);
      }
    } else {
      tree_.insertPointCloud(pc, origin, max_range_, true, true);
    }
    tree_.updateInnerOccupancy();
    tree_.prune();

    //publishing the octomap tree
    octomap_msgs::Octomap oct_msg;
    octomap_msgs::binaryMapToMsg(tree_,oct_msg);
    oct_msg.header.stamp=ros::Time::now();
    oct_msg.header.frame_id=map_frame_;

    pub_octree.publish(oct_msg);

    //publishing rviz
    publishMarkers();
  }

  void publishMarkers() {
    std::map<double, std::vector<geometry_msgs::Point> > buckets;
    for (auto it = tree_.begin_leafs(), end = tree_.end_leafs(); it != end; ++it) {
      if (!tree_.isNodeOccupied(*it)) continue;
      geometry_msgs::Point p;
      p.x = it.getX(); p.y = it.getY(); p.z = it.getZ();
      buckets[it.getSize()].push_back(p);
    }

    visualization_msgs::MarkerArray arr;
    visualization_msgs::Marker del;
    del.header.frame_id = map_frame_;
    del.header.stamp = ros::Time::now();
    del.ns = "octomap";
    del.action = visualization_msgs::Marker::DELETEALL;
    arr.markers.push_back(del);

    int id = 0;
    for (auto b = buckets.begin(); b != buckets.end(); ++b) {
      visualization_msgs::Marker m;
      m.header.frame_id = map_frame_;
      m.header.stamp = ros::Time::now();
      m.ns = "octomap";
      m.id = id++;
      m.type = visualization_msgs::Marker::CUBE_LIST;
      m.action = visualization_msgs::Marker::ADD;
      m.scale.x = m.scale.y = m.scale.z = b->first;
      m.color.r = 0.1; m.color.g = 0.6; m.color.b = 1.0; m.color.a = 1.0;
      m.pose.orientation.w = 1.0;
      m.points = b->second;
      arr.markers.push_back(m);
    }
    pub_rviz.publish(arr);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "mapping_node");
  ros::NodeHandle nh;
  Mapper node(nh);
  ros::spin();
}