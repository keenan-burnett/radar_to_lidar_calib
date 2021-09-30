import os
import os.path as osp
import rosbag
import numpy as np
import argparse
from cv_bridge import CvBridge
import cv2
from sensor_msgs import point_cloud2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--bagid')
    args = parser.parse_args()
    root = args.root
    radar_root = osp.join(root, 'radar')
    lidar_root = osp.join(root, 'lidar')
    if not osp.isdir(radar_root):
        os.mkdir(radar_root)
    if not osp.isdir(lidar_root):
        os.mkdir(lidar_root)
    topics = ['/velodyne_points', '/talker1/Navtech/Polar', '/navsat/odom']
    
    bagfiles = sorted([f for f in os.listdir(root) if '.bag' in f and args.bagid in f])
    v = 1
    for bag in bagfiles:
        bag = rosbag.Bag(osp.join(root, bag))
        for topic, msg, t in bag.read_messages(topics):
            timestamp = int(msg.header.stamp.to_nsec() * 1e-3)
            if topic == '/navsat/odom':
                vel = msg.twist.twist.linear
                v = np.sqrt(vel.x**2 + vel.y**2)
            if topic == '/velodyne_points':
                if v > 0.1:
                    continue
                cloud_points = list(point_cloud2.read_points(msg, skip_nans=True))
                points = np.array(cloud_points, dtype=np.float32)
                points.astype(np.float32).tofile(osp.join(lidar_root, '{}.bin'.format(timestamp)))
            if topic == '/talker1/Navtech/Polar':
                if v > 0.1:
                    continue
                img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="mono8")
                cv2.imwrite(osp.join(radar_root, '{}.png'.format(timestamp)), img)
