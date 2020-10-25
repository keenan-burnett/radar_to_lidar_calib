# radar_to_lidar_calib
Calculates the extrinsic calibration between a Navtech radar and a 3D (Velodyne) lidar

# Instructions
The robot/vehicle should be outdoors and stationary during the calibration process.

It is expected that Navtech radar data is being published as a Polar image using the format described by the Oxford Radar Robotcar Dataset. This format expects that the polar image is shift horizontally by 11 pixels such that timestamp and azimutm data can be baked into the image instead of being published on a separate topic. The first eight pixels correspond to the UNIX timestamp as int64 (cols 0-7). Columns 8-9 encode the sweep counter as uint16, converted to angle in radians with angle = pi * sweep_counter / 2800. Column 10 represents a valid flag, which we do not use but preserve for compatibility with their format.

Collect at least one pair of radar and lidar data extracted using the extract.py script.
This script expects that your radar and lidar topics are being published in ROS.

Collecting more pairs of data in different locations will lead to a more accurate final result since we average the translation and rotation estimates over the number of pairs.

Note: this calibration is mostly useful for estimating rotation. I've verified that with 10 radar-lidar pairs, I was able to get a rotation accuracy of < 0.1 degrees. The translation estimation isn't as good, and probably isn't usable. It might be possible to upsample the cartesian image on a cropped region to get a better translation estimate, but it would be tough to beat hand measurements or a CAD model (1 cm error).