#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import tf
from RoboHandler import RoboHandler
robo = None
import numpy as np

def callback(data):
    global robo
    if data.data == "init":
        if robo is None:
            robo = RoboHandler()
    elif data.data.startswith('set'):
        data_arr = data.data.split(" ")
        init_object = data_arr[1]
        if init_object.startswith('w'):
            init_object = 'wood_block'
        elif init_object.startswith('c'):
            init_object = 'cheezit'
        else:
            init_object = 'domino'
        robo.set_object(init_object)
    elif data.data == "order":
        grasps = robo.order_grasps()
        grasp_left_relative_pose = grasps['grasp_left_relative_pose']
        grasp_right_relative_pose = grasps['grasp_right_relative_pose']

        br = tf.TransformBroadcaster()

        # import IPython
        # IPython.embed()

        left_position = np.round(grasp_left_relative_pose[0]*2,3)
        left_direction = np.round(grasp_left_relative_pose[1],3)
        br.sendTransform((left_position[0], left_position[1], left_position[2]),
                     (left_direction[0], left_direction[1], left_direction[2], left_direction[3]),
                     rospy.Time.now(),
                     "l_gripper_tool_frame",
                     "object")
        
        right_position = np.round(grasp_right_relative_pose[0]*2,3)
        right_direction = np.round(grasp_right_relative_pose[1],3)
        br.sendTransform((right_position[0], right_position[1], right_position[2]),
                     (right_direction[0], right_direction[1], right_direction[2], right_direction[3]),
                     rospy.Time.now(),
                     "r_gripper_tool_frame",
                     "object")
        # print msg
        import IPython
        IPython.embed()
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('robo_handler_listener', anonymous=True)

    rospy.Subscriber("robo_handler", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()