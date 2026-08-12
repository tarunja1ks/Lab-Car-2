#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

class planning{
    ros::Subscriber endpoint_reader; 
    visualization_msgs::MarkerArray arr;
    public:
    planning(rosNodeHandler nh){
        endpoint_reader=nh.subscribe("/draw_path",1); 

    }
    void cb_planning_drawing(const std_msgs::String::ConstPtr endpoint_information){
        // make the 2 endpoints
        
        //use the model to draw the path in rviz
        

    }

    
}

int main(int argc, char** argv){

}