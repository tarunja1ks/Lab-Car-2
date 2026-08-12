#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/String.h>
#include <sstream>

using namespace std;
class planning{
    ros::Subscriber endpoint_reader; 
    ros::Publisher marker_pub;
    int marker_id=0;
    
    visualization_msgs::MarkerArray arr;
    std::string map_frame;
    
    public:
    planning(ros::NodeHandle& nh){
        endpoint_reader=nh.subscribe<std_msgs::String>("/draw_path",1,&planning::cb_planning_drawing,this); 
        marker_pub=nh.advertise<visualization_msgs::Marker>("/planning_spheres",2);
        map_frame="map";

    }
    void cb_planning_drawing(const std_msgs::String::ConstPtr& endpoint_information){
        // make the 2 endpoints
        stringstream ss(endpoint_information->data);
        double sx,sy,gx,gy;
        ss>>sx>>sy>>gx>>gy;
        plot_sphere(sx,sy);
        plot_sphere(gx,gy);
        
        
        //use the model to draw the path in rviz
        

    }

    void plot_sphere(double x, double y){
        visualization_msgs::Marker m;
        m.id=marker_id;
        marker_id+=1;
        m.type=visualization_msgs::Marker::SPHERE;
        m.pose.position.x=x;
        m.pose.position.y=y;
        m.pose.position.z=0.05;
        m.pose.orientation.w=1.0;
 
        m.scale.x = 0.4;
        m.scale.y = 0.4;
        m.scale.z = 0.4;

        m.color.r = 0.0f;
        m.color.g = 1.0f;
        m.color.b = 0.0f;
        m.color.a = 1.0;   
        m.header.frame_id=map_frame;

        marker_pub.publish(m);
    }

    
};

int main(int argc, char** argv){
    ros::init(argc, argv, "planning_node");
    ros::NodeHandle nh;
    planning node(nh);
    ros::spin();
}