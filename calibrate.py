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

def lidar_to_polar_image(pc, range_resolution, azimuth_resolution, range_bins, azimuth_bins):
    polar = np.zeros((azimuth_bins, range_bins))
    for i in range(0, pc.shape[1]):
        if x[2, i] < -1.5 or x[2, i] > 1.5:
            continue
        r = np.sqrt(x[0, i]**2 + x[1, i]**2)
        theta = np.arctan2(x[1, i], x[0, i])
        if theta < 0:
            theta += 2 * np.pi
        range_bin = int(r / range_resolution)
        azimuth_bin = int(theta / azimuth_resolution)
        if 0 < range_bin and range_bin < range_bins and 0 < azimuth_bin and azimuth_bin < azimuth_bins:
            polar[azimuth_bin, range_bin] = 255
    polar = np.flip(polar, axis=0)
    return polar

if __name__ == "__main__":
    datadir = "."
    radar_files = os.listdir(datadir + "/radar/")
    lidar_files = os.listdir(datadir + "/lidar/")
    lidar_files.sort()
    radar_files.sort()
    if not os.path.exists("figs"):
        os.makedirs("figs")
    min_range = 42
    radar_resolution = 0.0596
    range_bins = 3360
    azimuth_bins = 400
    cart_resolution = 0.0596 # 0.2384
    cart_pixel_width = 3356 # 838
    azimuth_step = np.pi / 200
    calibrate_translation = True
    upsample_azimuths = 2
    cart_res2 = cart_resolution # (>= radar_resolution) decrease for better translation estimation
    cart_width2 = int(200 / cart_res2)
    if upsample_azimuths > 1.0:
        azimuth_step = azimuth_step / upsample_azimuths
    visualize_results = True

    rotations = []
    translations = []

    for i in range(0, len(radar_files)):
        _, azimuths, _, fft_data, _ = load_radar(datadir + "/radar/" + radar_files[i])
        azimuth_bins = fft_data.shape[0]
        range_bins = fft_data.shape[1]
        if upsample_azimuths > 1.0:
            azimuths[1] = azimuth_step
            fft_data = cv2.resize(fft_data, dsize = (0, 0), fx = 1, fy = upsample_azimuths, interpolation = cv2.INTER_CUBIC)

        targets = cen2018features(fft_data)
        polar = targets_to_polar_image(targets, fft_data.shape)
        cart = radar_polar_to_cartesian(azimuths, polar, radar_resolution, cart_resolution, cart_pixel_width)
        cart = np.where(cart > 0, 255, 0)

        x = load_lidar(datadir + "/lidar/" + lidar_files[i])
        cart_lidar = lidar_to_cartesian_image(x, cart_pixel_width, cart_resolution)
        polar_lidar = lidar_to_polar_image(x, radar_resolution, azimuth_step, range_bins, azimuth_bins * upsample_azimuths)

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
        print('rotation index: {} rotation: {} radians, {} degrees'.format(rotation_index, rotation, rotation * 180 / np.pi))
        rotations.append(rotation)
        R = get_rotation(rotation) # Rotation to convert points in lidar frame to points in radar frame (R_12)
        xprime = x
        for j in range(0, x.shape[1]):
            xprime[:,j] = np.squeeze(np.matmul(R, x[:,j].reshape(3,1)))
        cart_lidar2 = lidar_to_cartesian_image(xprime, cart_pixel_width, cart_resolution)

        if calibrate_translation:
            cart1 = radar_polar_to_cartesian(azimuths, polar, radar_resolution, cart_res2, cart_width2)
            cart2 = lidar_to_cartesian_image(xprime, cart_width2, cart_res2)
            f1 = np.fft.fft2(cart1)
            f2 = np.fft.fft2(cart2)
            p = (f2 * f1.conjugate())
            p = p / abs(p)
            p = np.fft.ifft2(p)
            p = abs(p)
            delta_x = np.where(p == np.amax(p))[0][0]
            delta_y = np.where(p == np.amax(p))[1][0]
            if delta_x > cart_width2 / 2:
                delta_x -= cart_width2
            if delta_y > cart_width2 / 2:
                delta_y -= cart_width2
            delta_x *= cart_res2
            delta_y *= cart_res2
            xbar = np.array([delta_x, delta_y, 1]).reshape(3, 1)
            print('delta_x: {} delta_y: {}'.format(xbar[0], xbar[1]))
            translations.append(xbar.transpose())
            
    translations = np.array(translations)
    rotations = np.array(rotations)
    rotation = np.mean(rotations)
    translation = np.mean(translations, axis=0)
    print('rotation: {} radians, {} degrees'.format(rotation, rotation * 180 / np.pi))
    print('x: {} y : {}'.format(translation[0, 0], translation[0, 1]))

    if visualize_results:
        cart_resolution = 0.2384
        cart_pixel_width = 838
        azimuth_step = np.pi / 200
        azimuth_bins = 400
        R = get_rotation(rotation)
        for i in range(0, len(radar_files)):
            _, azimuths, _, fft_data, _ = load_radar(datadir + "/radar/" + radar_files[i])
            targets = cen2018features(fft_data)
            polar = targets_to_polar_image(targets, fft_data.shape)
            cart = radar_polar_to_cartesian(azimuths, polar, radar_resolution, cart_resolution, cart_pixel_width)
            cart = np.where(cart > 0, 255, 0)
            x = load_lidar(datadir + "/lidar/" + lidar_files[i])
            for j in range(0, x.shape[1]):
                x[:,j] = np.squeeze(np.matmul(R, x[:,j].reshape(3,1)))
            cart_lidar = lidar_to_cartesian_image(x, cart_pixel_width, cart_resolution)
            rgb = np.zeros((cart_pixel_width, cart_pixel_width, 3), np.uint8)
            rgb[..., 0] = cart_lidar
            rgb[..., 1] = cart
            cv2.imwrite("figs/combined" + str(i) + ".png", np.flip(rgb, axis=2))
            fig, axs = plt.subplots(1, 3, tight_layout=True)
            axs[0].imshow(cart, cmap=cm.gray)
            axs[1].imshow(cart_lidar, cmap=cm.gray)
            axs[2].imshow(rgb)
            plt.show()
            

