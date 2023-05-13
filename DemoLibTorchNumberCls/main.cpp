#include <iostream>
#include <opencv2/opencv.hpp>

#include "torch/script.h"

#include <fstream>


using namespace cv;
using namespace std;


int main(int argc, char *argv[]) {
    string model_path = "../model.jit";
    string img_path = argv[1];

    Mat img = cv::imread(img_path);
    
    auto tensor = torch::from_blob(img.data, {img.rows, img.cols, img.channels()}, at::kByte);

    // cout << tensor.shape << endl;
    tensor = tensor.permute({2, 0, 1});
    tensor = tensor.unsqueeze(0);

    tensor = tensor.toType(c10::kFloat).to(c10::DeviceType::CPU);

    // cout << tensor << endl;

    torch::jit::Module model = torch::jit::load(model_path);
    
    torch::NoGradGuard noGrad;

    // forward 
    torch::Tensor out = model({tensor}).toTensor();

    cout << "out: " << out << endl;

    int idx = torch::argmax(out, 1).item().toInt();
    
    cout << "result: " << idx << endl;

    return 0;
}





