import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import shift
from features import *
from radar import *

datadir = "."

def targets_to_polar_image(targets, shape):
    polar = np.zeros(shape)
    N = targets.shape[0]
    for i in range(0, N):
        polar[targets[i, 0], targets[i, 1]] = 255
    return polar

def get_rotation(theta):
    R = np.identity(3)
    R[0:2, 0:2] = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    return R

def load_lidar(filename):
    lidar = np.loadtxt(filename, delimiter=',') 
    return lidar[:, 0:3].transpose()

def lidar_to_cartesian_image(pc, cart_pixel_width, cart_resolution):
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    cart_img = np.zeros((cart_pixel_width, cart_pixel_width))
    for i in range(0, pc.shape[1]):
        if x[2, i] < -1.5 or x[2, i] > 1.5:
            continue
        u = int((cart_min_range - x[1, i]) / cart_resolution)
        v = int((cart_min_range - x[0, i]) / cart_resolution)
        if 0 < u and u < cart_pixel_width and 0 < v and v < cart_pixel_width:
            cart_img[v, u] = 255
    return cart_img

if __name__ == "__main__":
    if len(sys.argv) > 1:
        datadir = sys.argv[1]
    radar_files = os.listdir(datadir + "/radar/")
    lidar_files = os.listdir(datadir + "/lidar/")
    lidar_files.sort()
    radar_files.sort()

    min_range = 42
    radar_resolution = 0.0596
    cart_resolution = 0.2384
    cart_pixel_width = 838
    azimuth_step = np.pi / 200

    for i in range(0, len(radar_files)):
        timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(datadir + "/radar/" + radar_files[i])
        targets = cen2018features(fft_data)
        polar = targets_to_polar_image(targets, fft_data.shape)
        cart = radar_polar_to_cartesian(azimuths, polar, radar_resolution, cart_resolution, cart_pixel_width)
        cart = np.where(cart > 0, 255, 0)
        # for row in range(0, cart.shape[0]):
        #     for col in range(0, cart.shape[1]):
        #         if cart[row, col] > 0:
        #             cart(row, col) = 255
        
        # polar2 = np.roll(polar2, -rotation_index, axis=0)
        # cart2 = radar_polar_to_cartesian(azimuths, polar2, radar_resolution, cart_resolution, cart_pixel_width)

        # f1 = np.fft.fft2(cart)
        # f2 = np.fft.fft2(cart2)
        # p = (f2 * f1.conjugate())
        # p = p / abs(p)
        # p = np.fft.ifft2(p)
        # p = abs(p)
        # delta_x = np.where(p == np.amax(p))[0][0]
        # delta_y = np.where(p == np.amax(p))[0][0]
        # xbar = np.array([delta_x, delta_y]).reshape(2, 1)
        # R = get_rotation(-rotation)
        # xbar = np.matmul(R, xbar)
        # print('delta_x: {} delta_y: {}'.format(xbar[0], xbar[1]))

        x = load_lidar(datadir + "/lidar/" + lidar_files[i])
        cart_lidar = lidar_to_cartesian_image(x, cart_pixel_width, cart_resolution)
        polar_lidar = cartesian_to_polar(cart_lidar, radar_resolution, azimuth_step, 3360, 400, cart_pixel_width, cart_resolution)

        f1 = np.fft.fft2(polar)
        f2 = np.fft.fft2(polar_lidar)
        p = (f2 * f1.conjugate())
        p = p / abs(p)
        p = np.fft.ifft2(p)
        p = abs(p)
        rotation_index = np.where(p == np.amax(p))[0][0]
        rotation = rotation_index * azimuth_step

        if rotation > np.pi:
            rotation = 2 * np.pi - rotation
        R = get_rotation(rotation)
        xprime = x
        for j in range(0, x.shape[1]):
            xprime[:,j] = np.squeeze(np.matmul(R, x[:,j].reshape(3,1)))

        cart_lidar2 = lidar_to_cartesian_image(xprime, cart_pixel_width, cart_resolution)
        rgb = np.zeros((cart_pixel_width, cart_pixel_width, 3), np.uint8)
        rgb[..., 0] = cart_lidar2
        rgb[..., 1] = cart
        cv2.imwrite("radar.png", cart)
        cv2.imwrite("lidar.png", cart_lidar)
        cv2.imwrite("combined.png", np.flip(rgb, axis=2))
        fig, axs = plt.subplots(1, 3, tight_layout=True)
        axs[0].imshow(cart, cmap=cm.gray)
        axs[1].imshow(cart_lidar, cmap=cm.gray)
        axs[2].imshow(rgb)
        plt.show()

        print('rotation index: {} rotation: {} radians, {} degrees'.format(rotation_index, rotation, rotation * 180 / np.pi))
        print(R)
        print(x[:, 0].reshape(3,1))
        print(np.matmul(R, x[:, 0].reshape(3,1)))


    