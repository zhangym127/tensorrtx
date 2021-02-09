#!/bin/bash

set -e

cd ./build
rm -r *
cmake ..
make -j12
#cd ../

#./yolov5 -i imageFolder -o outputFolder



