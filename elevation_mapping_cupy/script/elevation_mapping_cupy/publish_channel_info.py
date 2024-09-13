#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from elevation_map_msgs.msg import ChannelInfo


class ChannelInfoPub():
    def __init__(self):
        self.timer = rospy.Timer(rospy.Duration(1/15), self.publish_channel_info)
        self.channel_info_pub = rospy.Publisher('/zed2/zed_node/rgb/channel_info', ChannelInfo, queue_size=1)
        
        channel_info = ChannelInfo()
        channel_info.channels = ['r', 'g', 'b', 'a']
        self.channel_info = channel_info

    def publish_channel_info(self, event):
        self.channel_info.header.stamp = rospy.Time.now()
        self.channel_info_pub.publish(self.channel_info)

if __name__ == '__main__':
    rospy.init_node('channel_info_pub')
    channel_info_pub = ChannelInfoPub()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            print("Error")
