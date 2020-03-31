#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>
#include <map>
#include <functional>
#include <string>

using namespace cv;
using namespace std;

void mean_kernel(Mat &kernel, int ksize);
void gaussian_kernel(Mat &kernel, int ksize, double sigma=0.849);

vector<int> ksizes {3, 5, 9};
vector<string> smooth_kernel_names {"Mean", "Gauss"};
vector<function<void(Mat&, int)>> smooth_kernel_fun {mean_kernel, bind(gaussian_kernel, placeholders::_1, placeholders::_2, 1.)};

vector<string> diff_kernel_names {"Laplace", "Sobel", "Robert",};
vector<vector<char>> kernel_weights0 {{0,-1,0,-1,4,-1,0,-1,0}, {-1,-2,-1,0,0,0,1,2,1}, {0,0,0,0,-1,0,0,0,1}};
vector<vector<char>> kernel_weights1 {{0,-1,0,-1,4,-1,0,-1,0}, {-1,0,1,-2,0,2,-1,0,1}, {0,0,0,0,0,-1,0,1,0}};

void mean_kernel(Mat &kernel, int ksize)
{
    kernel = Mat::ones(ksize, ksize, CV_32F) / (float)(ksize * ksize);
}

void gaussian_kernel(Mat &kernel, int ksize, double sigma)
{
    kernel.create(ksize, ksize, CV_32F);
    auto fun = [=](float x, float y) -> float {
        return exp((-x*x-y*y) / (2*sigma*sigma))/ (2 * CV_PI * sigma * sigma);};

    for (int i = 0; i < ksize; ++i) {
        auto p = kernel.ptr<float>(i);
        for (int j = 0; j < ksize; ++j) {
            p[j] = fun(j-(ksize-1)/2, i-(ksize-1)/2);
            // p[j] = fun(j-ksize/2, i-ksize/2);
        }
    }

    auto weight_sum = cv::sum(kernel);
    cv::divide(kernel, Mat(ksize, ksize, kernel.type(), weight_sum), kernel);
}

int main(int argc, char const *argv[])
{
    const char *image_path = (argc > 1) ? argv[1] : "img_test.jpg";
    Mat img, image_color, image_gray, image_dst;
    img = imread(image_path, IMREAD_COLOR);
    if (img.empty()) {
        cout << "Cannot read image: " << image_path << std::endl;
        return -1;
    }

    namedWindow("image", CV_WINDOW_AUTOSIZE);

    image_color = img;
    cvtColor(image_color, image_gray, CV_BGR2GRAY);

    map<string, Mat> images {{"image_gray", image_gray}, {"image_color", image_color}};


    for (auto &item : images) {
        auto image_name = item.first;
        auto image = item.second;

        cout << endl << endl;
        cout << image_name << endl;
        imshow("image", image);
        waitKey();

        cout << endl;
        for (size_t i = 0; i < smooth_kernel_names.size(); ++i) {
            cout << smooth_kernel_names[i] << endl;

            for (size_t j = 0; j < ksizes.size(); ++j) {
                Mat kernel, image_diff;

                smooth_kernel_fun[i](kernel, ksizes[j]);

                cout << smooth_kernel_names[i] << "\tsize:" << ksizes[j] << "\tsmooth" << endl;
                filter2D(image, image_dst, image.depth(), kernel);
                imshow("image", image_dst);
                waitKey();

                cout << smooth_kernel_names[i] << "\tsize:" << ksizes[j] << "\tdiff" << endl;
                subtract(image, image_dst, image_diff);
                imshow("image", image_diff);
                waitKey();

                cout << smooth_kernel_names[i] << "\tsize:" << ksizes[j] << "\tsharpen" << endl;
                add(image, image_diff, image_dst);
                imshow("image", image_dst);
                waitKey();
            }
        }

        cout << endl;
        for (size_t i = 0; i < diff_kernel_names.size(); ++i) {
            cout << diff_kernel_names[i] << endl;

            Mat diff, diff0, diff1, abs_diff0, abs_diff1;

            filter2D(image, diff0, image.depth(), Mat(3,3,CV_8S,kernel_weights0[i].data()));
            filter2D(image, diff1, image.depth(), Mat(3,3,CV_8S,kernel_weights1[i].data()));
            convertScaleAbs(diff0, abs_diff0);
            convertScaleAbs(diff1, abs_diff1);
            addWeighted(abs_diff0, 0.5, abs_diff1, 0.5, 0, diff);

            cout << diff_kernel_names[i] + "\tdiff" << endl;
            imshow("image", diff);
            waitKey();

            cout << diff_kernel_names[i] + "\tsharpen" << endl;
            add(image, diff, image_dst);
            imshow("image", image_dst);
            waitKey();
        }
    }

    return 0;
}
