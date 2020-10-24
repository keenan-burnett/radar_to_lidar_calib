import rospy
import numpy as np
import message_filters
import cv2
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge, CvBridgeError

index = 0

def callback(lidar_msg, radar_msg):
    global index
    cloud_points = list(point_cloud2.read_points(lidar_msg, skip_nans=True))
    points = np.array(cloud_points, dtype=np.float32)
    np.savetxt('lidar/{}.txt'.format(index), points, delimiter=',')
    img = CvBridge().imgmsg_to_cv2(radar_msg, "mono8")
    cv2.imwrite('radar/{}.png'.format(index), img)
    print(index)
    index += 1
    
if __name__ == "__main__":
    rospy.init_node('radar_lidar_extract')
    lidar_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)
    radar_sub = message_filters.Subscriber('/talker1/Navtech/Polar', Image)
    ts = message_filters.ApproximateTimeSynchronizer([lidar_sub, radar_sub], 100, 0.1)
    ts.registerCallback(callback)
    rospy.spin()
