#!/bin/bash
pip install gdown
gdown https://drive.google.com/uc?id=1tTFq98z2bj1Ya0seSob6I6tJcp82rc1i -O data.zip
unzip data.zip
mkdir lidar
mkdir radar
mv radar_sample_data/*.png radar
mv radar_sample_data/*.txt lidar
rm -rf radar_sample_data
rm data.zip
