#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "yolov5.hpp"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

#define NET s  // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static IRuntime* runtime;
static ICudaEngine* engine;
static IExecutionContext* context;
static int inputIndex;
static int outputIndex;
static void* buffers[2];
static cudaStream_t stream;

static void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

YOLOv5::YOLOv5(std::string & engine_name) {
	/* 设置推理设备，CPU或GPU */
	cudaSetDevice(DEVICE);
	
	/* 读取引擎文件 */
	char *trtModelStream{ nullptr };
	size_t size{ 0 };
	std::ifstream file(engine_name, std::ios::binary);
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
	}

	/* 创建推理引擎 */
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CHECK(cudaStreamCreate(&stream));
}

YOLOv5::~YOLOv5() {
	// Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void YOLOv5::detect(std::string image_name, std::vector<Yolo::Detection> & output) {
	
	/* 打印待推理的图像文件 */
	std::cout << image_name << std::endl;
	
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];

	/* 把一个batch的图像文件加载到内存中——data */
	cv::Mat img = cv::imread(image_name);
	if (img.empty()) return;
	cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
	int i = 0;
	/* 对RGB三个通道的图像数据进行映射，int8转成float */
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = pr_img.data + row * pr_img.step;
		for (int col = 0; col < INPUT_W; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}

	/* 开始推理 */
	auto start = std::chrono::system_clock::now();
	doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
	auto end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	
	/* NMS非极大值抑制 */
	auto & res = output;
	nms(res, &prob[0], CONF_THRESH, NMS_THRESH);
}

void YOLOv5::draw(std::string image_dir, 
				  std::string image_name, 
				  std::vector<Yolo::Detection> & input, 
				  std::string output_dir) {
	/* 把检测框标注在图像上 */
	auto & res = input;
	//std::cout << res.size() << std::endl;
	cv::Mat img = cv::imread(image_dir + image_name);
	for (size_t j = 0; j < res.size(); j++) {
		cv::Rect r = get_rect(img, res[j].bbox);
		cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
		cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
	}
	cv::imwrite(output_dir + image_name, img);
}

