
#include <fstream>
#include <dirent.h>
#include "yolov5.hpp"


static int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

int main(int argc, char** argv) {

	/* 给出引擎文件名称，引擎文件发生变化时需要修改 */
    std::string engine_name = "yolov5s.engine";
	
	/* 创建YOLOv5对象指针 */
	YOLOv5 * yolov5 = new YOLOv5(engine_name);

	/* 给出输入和输出文件夹的名称 */
	std::string images_folder = "/dockerData/jason/data/mask";
	std::string output_folder = "./";
	
	/* 给出输出结果文件的名称 */
	std::string result_name = "result.txt";
	
	/* 检查输入参数 */
	if (argc != 1 && argc != 3 && argc != 5) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -i imgFolder -o outFolder" << std::endl;
        return -1;
    }
	
	/* 从命令行读取输入和输出文件夹 */
	for (int i = 1; i < argc; i++) {
		if (std::string(argv[i]) == "-i") {
			images_folder = argv[i + 1];
			i++;
			continue;
		}
		if (std::string(argv[i]) == "-o") {
			output_folder = argv[i + 1];
			i++;
			continue;
		}
	}
	
	/* 打开输出结果文件 */
	std::ofstream result;//创建文件
    result.open(output_folder + "/" + result_name);

	/* 从指定的文件夹中读取需要推理的图像文件 */
    std::vector<std::string> file_names;
    if (read_files_in_dir(images_folder.c_str(), file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

	/* 遍历每个图像文件 */
    for (int f = 0; f < (int)file_names.size(); f++) {
		std::vector<Yolo::Detection> objs;
		/* 推理 */
		yolov5->detect(images_folder + "/" + file_names[f], objs);
		/* 在图像上画出框 */
		yolov5->draw(images_folder + "/", file_names[f], objs, output_folder);
		/* 统计样本个数 */
		int face = 0;
		int maks = 0;
		int help = 0;
		int unclr = 0;
		for (size_t i = 0; i < objs.size(); i++)
		{
			if((objs[i].class_id == 0) && (objs[i].conf>0.1)) {
				face++;
			}
			else if ((objs[i].class_id == 1) && (objs[i].conf>0.1)) {
				maks++;
			}
			else if ((objs[i].class_id == 2) && (objs[i].conf>0.1)) {
				help++;
			}
			else if ((objs[i].class_id == 3) && (objs[i].conf>0.1)) {
				unclr++;
			}
		}
		result << images_folder + "/" + file_names[f] 
			   << " " << face << " " << maks << " " << help << " " << unclr << std::endl; 
    }
	
	/* 删除对象并返回 */
	result.close();
	delete yolov5;
    return 0;
}
