#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/String.h>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>

using namespace std;


struct Mat {
    int rows = 0, cols = 0;
    vector<double> v;
    double at(int r, int c) const { return v[r * cols + c]; }
};


struct NTField {
    map<string, Mat> w;

    bool load(const string& path) {
        ifstream f(path);
        if (!f) return false;
        int n = 0;
        f >> n;
        for (int i = 0; i < n; ++i) {
            string name; int r, c;
            if (!(f >> name >> r >> c)) return false;
            Mat m; m.rows = r; m.cols = c; m.v.resize((size_t)r * c);
            for (size_t k = 0; k < m.v.size(); ++k) f >> m.v[k];
            w[name] = m;
        }
        const char* need[] = {"f.layers.0.weight","f.layers.0.bias",
                              "f.layers.2.weight","f.layers.2.bias",
                              "tao.layers.0.weight","tao.layers.0.bias",
                              "tao.layers.2.weight","tao.layers.2.bias"};
        for (const char* k : need) if (!w.count(k)) return false;
        return true;
    }

    static double softplus(double x) {
        return x > 20.0 ? x : std::log1p(std::exp(x));
    }
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // y = W x + b
    vector<double> linear(const vector<double>& x,
                          const string& wk, const string& bk) const {
        const Mat& W = w.at(wk);
        const Mat& b = w.at(bk);
        vector<double> y(W.rows, 0.0);
        for (int i = 0; i < W.rows; ++i) {
            double s = b.v[i];
            for (int j = 0; j < W.cols; ++j) s += W.at(i, j) * x[j];
            y[i] = s;
        }
        return y;
    }

    vector<double> encode(double x, double y) const {
        vector<double> h = linear({x, y}, "f.layers.0.weight", "f.layers.0.bias");
        for (double& v : h) v = softplus(v);
        h = linear(h, "f.layers.2.weight", "f.layers.2.bias");
        for (double& v : h) v = softplus(v);
        return h;
    }

    double tau(const vector<double>& fa, const vector<double>& fb) const {
        vector<double> s(fa.size() * 2);
        for (size_t i = 0; i < fa.size(); ++i) {
            s[i]             = std::max(fa[i], fb[i]);
            s[fa.size() + i] = std::min(fa[i], fb[i]);
        }
        vector<double> h = linear(s, "tao.layers.0.weight", "tao.layers.0.bias");
        for (double& v : h) v = softplus(v);
        h = linear(h, "tao.layers.2.weight", "tao.layers.2.bias");
        return sigmoid(h[0]);
    }

    // arrival time from (x,y) to the goal
    double T(double x, double y, const vector<double>& fg,
             double gx, double gy) const {
        double dx = gx - x, dy = gy - y;
        double r = std::sqrt(dx * dx + dy * dy + 1e-12);
        double t = tau(encode(x, y), fg);
        return r / std::max(t, 1e-8);
    }
};

class planning {
    ros::Subscriber endpoint_reader;
    ros::Publisher marker_pub;
    int marker_id = 0;

    visualization_msgs::MarkerArray arr;
    std::string map_frame;

    NTField net;
    bool have_net = false;
    double step_size, goal_tol, fd_h;
    int max_steps;

public:
    planning(ros::NodeHandle& nh) {
        ros::NodeHandle pnh("~");
        std::string weights;
        pnh.param<std::string>("weights", weights, "ntfield_weights.txt");
        pnh.param("step_size", step_size, 0.05);
        pnh.param("goal_tol", goal_tol, 0.15);
        pnh.param("fd_h", fd_h, 0.01);
        pnh.param("max_steps", max_steps, 4000);

        have_net = net.load(weights);
        if (!have_net) ROS_ERROR("could not load weights: %s", weights.c_str());

        endpoint_reader=nh.subscribe<std_msgs::String>("/draw_path",1,&planning::cb_planning_drawing,this);
        marker_pub=nh.advertise<visualization_msgs::Marker>("/planning_spheres",1);
        map_frame="map";
    }

    void cb_planning_drawing(const std_msgs::String::ConstPtr& endpoint_information){
        // make the 2 endpoints
        stringstream ss(endpoint_information->data);
        double sx,sy,gx,gy;
        ss>>sx>>sy>>gx>>gy;
        plot_sphere(sx,sy);
        plot_sphere(gx,gy);

        if (!have_net) return;
        std::vector<geometry_msgs::Point> path = solve(sx, sy, gx, gy);
        plot_path(path);
    }

    // follow -grad T from start to goal; grad by central differences
    std::vector<geometry_msgs::Point> solve(double sx, double sy,
                                            double gx, double gy) {
        std::vector<double> fg = net.encode(gx, gy);
        std::vector<geometry_msgs::Point> pts;
        double x = sx, y = sy;
        for (int i = 0; i < max_steps; ++i) {
            geometry_msgs::Point p; p.x = x; p.y = y; p.z = 0.05;
            pts.push_back(p);

            if (std::hypot(gx - x, gy - y) < goal_tol) break;

            double tx = (net.T(x + fd_h, y, fg, gx, gy) - net.T(x - fd_h, y, fg, gx, gy)) / (2 * fd_h);
            double ty = (net.T(x, y + fd_h, fg, gx, gy) - net.T(x, y - fd_h, fg, gx, gy)) / (2 * fd_h);

            double n = std::sqrt(tx * tx + ty * ty);
            if (n < 1e-9) break;                       // flat field, nowhere to go
            x -= step_size * tx / n;
            y -= step_size * ty / n;
        }
        geometry_msgs::Point g; g.x = gx; g.y = gy; g.z = 0.05;
        pts.push_back(g);
        return pts;
    }

    void plot_path(const std::vector<geometry_msgs::Point>& pts) {
        visualization_msgs::Marker m;
        m.header.frame_id = map_frame;
        m.header.stamp = ros::Time::now();
        m.ns = "ntfield_path";
        m.id = 0;                                  // fixed id: replaces previous path
        m.type = visualization_msgs::Marker::LINE_STRIP;
        m.action = visualization_msgs::Marker::ADD;
        m.pose.orientation.w = 1.0;
        m.scale.x = 0.05;
        m.color.r = 1.0f; m.color.g = 0.4f; m.color.b = 0.0f; m.color.a = 1.0;
        m.points = pts;
        marker_pub.publish(m);
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
