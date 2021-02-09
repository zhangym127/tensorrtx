
#ifndef YOLOV5_H
#define YOLOV5_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "yololayer.h"


class YOLOv5
{
public:
	YOLOv5(std::string & engine_name);
	~YOLOv5();

	void detect(std::string image_name, std::vector<Yolo::Detection> & output);
	void draw(std::string image_dir, 
			  std::string image_name, 
			  std::vector<Yolo::Detection> & input, 
			  std::string output_dir);
};



#endif //YOLOV5_H
