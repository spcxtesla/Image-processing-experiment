#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <iostream>

using namespace cv;
using namespace std;

void calc_hist(Mat &I, vector<int> &hist)
{
    CV_Assert(I.depth() == CV_8U);
    CV_Assert(I.channels() == 1);

    for (auto it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
        hist[*it]++;
    
    CV_DbgAssert(accumulate(hist.begin(), hist.end(), 0) == I.cols * I.rows);
}


void calc_hist(Mat &I, vector<float> &hist_prob)
{
    CV_Assert(I.depth() == CV_8U);
    CV_Assert(I.channels() == 1);

    int counts = I.cols * I.rows;
    vector<int> hist(256, 0);

    calc_hist(I, hist);

    for (int i = 0; i < 256; ++i)
        hist_prob[i] = (float)hist[i] / counts;

    CV_DbgAssert(accumulate(hist_prob.begin(), hist_prob.end(), 0) == 1);
}

void plot_poly(Mat &I, vector<float> points, const Scalar color)
{
    int width = I.cols, height = I.rows, pts_num = points.size();
    float max_y_point =  *max_element(points.begin(), points.end());

    for (size_t i = 1; i < points.size(); ++i)
        //Round operator should be done at last.
        line(I,
            Point(cvRound((float)(i-1) / pts_num * width), height - cvRound(points[i-1] / max_y_point * height)),
            Point(cvRound((float)(i) / pts_num * width), height - cvRound(points[i] /max_y_point * height)),
            color, 2, 8, 0);
}

void hist_equ_transform(vector<float> foo, Mat &lut)
{
    for (size_t i = 1; i < foo.size(); ++i)
        foo[i] += foo[i-1];
    uchar *p = lut.ptr();
    for (size_t i = 0; i < 256; ++i)
        p[i] = (uchar)(foo[i] * 255);
}

void hist_equ(Mat &src, Mat &dst, vector<float> &opdf, vector<float> &npdf)
{
    // calc_hist(src, opdf);
    Mat lut(1, 256, CV_8U);
    hist_equ_transform(opdf, lut);
    LUT(src, lut, dst);
    calc_hist(dst, npdf);
}

int main(int argc, char const *argv[])
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
    namedWindow("hist", CV_WINDOW_AUTOSIZE);

    vector<float> opdf(256, 0.0), npdf(256, 0.0);
    Mat hist_img(480, 640, CV_8UC3, Scalar(0, 0, 0));

    calc_hist(image_gray, opdf);
    plot_poly(hist_img, opdf, Scalar(255, 255, 255));
    imshow("hist", hist_img);
    imshow("image", image_gray);
    waitKey();

    hist_equ(image_gray, image_dst, opdf, npdf);
    plot_poly(hist_img, npdf, Scalar(128, 128, 128));
    imshow("hist", hist_img);
    imshow("image", image_dst);
    waitKey();

    Mat hist_image(480, 640, CV_8UC3, Scalar(0, 0, 0));

    vector<Mat> bgr_planes;
    split(image, bgr_planes);
    vector<vector<float>> opdfs(3, vector<float>(256, 0.0));
    auto npdfs = opdfs;

    for (size_t i = 0; i < opdfs.size(); ++i) {
        calc_hist(bgr_planes[i], opdfs[i]);
        plot_poly(hist_image,opdfs[i],Scalar(255*(0==i),255*(1==i),255*(2==i)));
    }
    imshow("hist", hist_image);
    imshow("image", image);
    waitKey();

    for (size_t i = 0; i < npdfs.size(); ++i) {
        hist_equ(bgr_planes[i], bgr_planes[i], opdfs[i], npdfs[i]);
        plot_poly(hist_image,npdfs[i],Scalar(255*(0==i),255*(1==i),255*(2==i)));
    }
    merge(bgr_planes, image_dst);
    imshow("hist", hist_image);
    imshow("image", image_dst);
    waitKey();

    return 0;
}
