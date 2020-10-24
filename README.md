# radar_to_lidar_calib
Calculates the extrinsic calibration between a Navtech radar and a 3D (Velodyne) lidar

# Instructions
The robot/vehicle should be outdoors and stationary during the calibration process.

It is expected that Navtech radar data is being published as a Polar image using the format described by the Oxford Radar Robotcar Dataset. This format expects that the polar image is shift horizontally by 11 pixels such that timestamp and azimutm data can be baked into the image instead of being published on a separate topic. The first eight pixels correspond to the UNIX timestamp as int64 (cols 0-7). Columns 8-9 encode the sweep counter as uint16, converted to angle in radians with angle = pi * sweep_counter / 2800. Column 10 represents a valid flag, which we do not use but preserve for compatibility with their format.

