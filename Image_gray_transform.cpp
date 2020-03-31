#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <iostream>
#include <functional>

using namespace cv;
using namespace std;

void apply_LUT(Mat &src, Mat &dst, function<uchar(uchar)> pf)
{
    CV_Assert(src.depth() == CV_8U);

    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for(int i = 0; i < 256; ++i)
        p[i] = pf(i);
    LUT(src, lookUpTable, dst);
    imshow("image", dst);
    waitKey(0);
}
void threshold_transform(Mat &src, Mat &dst, uchar thre)
{
    auto fun = [=](uchar r) -> uchar {return ((r > thre) ? 1 : 0) * 255;};
    apply_LUT(src, dst, fun);
}

// $s=cr^\gamma \qquad r\in[0,1]$
void gamma_transform(Mat &src, Mat &dst, float g=2, float c = 1.0)
{
    auto fun = [=](uchar r) -> uchar {return (c * pow(r/255., g)*255);};
    apply_LUT(src, dst, fun);
}

// $s=c\log_{v+1}(1+vr)\qquad r\in[0,1]$
void log_transform(Mat &src, Mat &dst, float v=1, float c=1)
{
    auto fun = [=](uchar r) -> uchar {
        return (c * 255. * log(1. + v*r/255.) / log(v + 1));};
    apply_LUT(src, dst, fun);
}

void inv_transform(Mat &src, Mat &dst)
{
    auto fun = [=](uchar r) -> uchar {return (255 - r);};
    apply_LUT(src, dst, fun);
}

int main (int argc, char **argv)
{
    String image_name((argc>1)?(argv[1]):("img_test.jpg"));
    Mat image, image_gray, image_dst;
    image = imread(image_name, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Cannot read image: " << image_name << std::endl;
        return -1;
    }   
    cvtColor(image, image_gray, CV_BGR2GRAY);

    namedWindow("image", CV_WINDOW_AUTOSIZE);
    cout << "image origin" << endl;
    imshow("image", image);
    waitKey(0);

    cout << "image gray" << endl;
    imshow("image", image_gray);
    waitKey(0);

    cout << "image threshold" << endl;
    // threshold(image_gray, image_dst, 128, 255, THRESH_BINARY);
    threshold_transform(image_gray, image_dst, 128);

    cout << "image log" << endl;
    log_transform(image_gray, image_dst, 7);

    cout << "image gamma" << endl;
    gamma_transform(image_gray, image_dst, 9);

    cout << "image inv" << endl;
    inv_transform(image, image_dst);

    return 0;
}
