#!/usr/bin/env python3

import rospy
import tf

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

ROBOT_NAME = "mushr"


class ground_truth:

    def __init__(self):
        rospy.init_node("ground_truth")

        self.pub = rospy.Publisher(
            "/ground_truth_pose",
            Pose2D,
            queue_size=1)

        self.odom_pub = rospy.Publisher(
            "/odom",
            Odometry,
            queue_size=10)

        self.br = tf.TransformBroadcaster()

        rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.cb)

    def cb(self, msg):

        if ROBOT_NAME not in msg.name:
            return

        i = msg.name.index(ROBOT_NAME)

        pose = msg.pose[i]
        twist = msg.twist[i]

        q = pose.orientation
        _, _, yaw = euler_from_quaternion(
            [q.x, q.y, q.z, q.w])

        gt = Pose2D()
        gt.x = pose.position.x
        gt.y = pose.position.y
        gt.theta = yaw
        self.pub.publish(gt)

        self.br.sendTransform(
            (pose.position.x,
             pose.position.y,
             pose.position.z),
            (q.x,
             q.y,
             q.z,
             q.w),
            rospy.Time.now(),
            "base_footprint",
            "odom"
        )

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose = pose
        odom.twist.twist = twist

        self.odom_pub.publish(odom)


if __name__ == "__main__":
    ground_truth()
    rospy.spin()
