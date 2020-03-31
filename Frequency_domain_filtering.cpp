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

void fftshift(const Mat &inputImg, Mat &outputImg)
{
    outputImg = inputImg.clone();
    outputImg = outputImg(Rect(0, 0, outputImg.cols & -2, outputImg.rows & -2));
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    Mat q0(outputImg, Rect(0, 0, cx, cy));
    Mat q1(outputImg, Rect(cx, 0, cx, cy));
    Mat q2(outputImg, Rect(0, cy, cx, cy));
    Mat q3(outputImg, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void calcPSD(const Mat& inputImg, Mat& outputImg, bool need_vis=true)
{
    Mat planes[2] = {Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);            // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
    if (need_vis) {
        planes[0].at<float>(0) = 0;         //set F(0,0)=0 which is also called DC for visualization better
        planes[1].at<float>(0) = 0;
    }
    // compute the PSD = sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)^2
    Mat imgPSD;
    magnitude(planes[0], planes[1], imgPSD);        //now imgPSD = sqrt(Power spectrum density)
    pow(imgPSD, 2, imgPSD);                         //it needs ^2 in order to get PSD
    outputImg = imgPSD;
    // logPSD = log(1 + PSD)
    if (need_vis) {
        Mat imglogPSD;
        imglogPSD = imgPSD + Scalar::all(1);
        log(imglogPSD, imglogPSD);
        outputImg = imglogPSD;
    }
}

void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_SCALE);

    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);

    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}

void filter2DFreq_vis(const Mat& inputImg, Mat& outputImg, const Mat& realH)
{

    namedWindow("nimage", CV_WINDOW_AUTOSIZE);
    namedWindow("npsd", CV_WINDOW_AUTOSIZE);
    namedWindow("filter");

    Mat H = realH;
    fftshift(H, H);
    filter2DFreq(inputImg, outputImg, H);
    normalize(outputImg, outputImg, 0, 1, NORM_MINMAX);

    Mat nimgPSD;
    calcPSD(outputImg, nimgPSD);
    fftshift(nimgPSD, nimgPSD);
    normalize(nimgPSD, nimgPSD, 0, 1, NORM_MINMAX);

    // if (1) {
    //     nimgPSD.convertTo(nimgPSD, CV_8UC1, 255);
    //     applyColorMap(nimgPSD, nimgPSD, COLORMAP_JET);
    // }
    
    imshow("nimage", outputImg);
    imshow("npsd", nimgPSD);
    imshow("filter", realH);
    waitKey();
    destroyWindow("filter");
    destroyWindow("nimage");
    destroyWindow("npsd");
}

void apply_ideal_filter(const Mat & inputImg, Mat &outputImg, float D0, bool lp)
{
    Mat H = Mat(inputImg.size(), inputImg.type(), Scalar(1*!lp));
    Point center(inputImg.cols/2, inputImg.rows/2);
    circle(H, center, D0, 1*lp, -1);

    cout << "ideal filter\t" << (lp?"low pass":"high pass") << endl;
    filter2DFreq_vis(inputImg, outputImg, H);
}

// $H(u,v)=\frac{1}{1+[D(u,v)/D_0]^{2n}}$
void apply_butterworth_filter(const Mat & inputImg, Mat &outputImg, float D0, float n, bool lp)
{
    Mat H;
    Point center(inputImg.cols/2, inputImg.rows/2);

    Mat tmp(inputImg.size(), CV_8UC1, 1), dist_map, tmp1;
    tmp.at<uchar>(center) = 0;
    distanceTransform(tmp, dist_map, DIST_L2, DIST_MASK_PRECISE, CV_32F);
    pow((dist_map / D0), 2*n, tmp1);
    divide(1, tmp1 + 1, H);
    
    if (!lp) H = 1 - H;

    cout << "butterworth filter\t" << (lp?"low pass":"high pass") << endl;
    filter2DFreq_vis(inputImg, outputImg, H);
}

int main(int argc, char const *argv[])
{
    const char *image_path = (argc > 1) ? argv[1] : "img_test.jpg";
    Mat img, image_dst;
    img = imread(image_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Cannot read image: " << image_path << std::endl;
        return -1;
    }

    namedWindow("image", CV_WINDOW_AUTOSIZE);
    namedWindow("psd", CV_WINDOW_AUTOSIZE);

    Mat image = img(Rect(0, 0, img.cols & -2, img.rows & -2));
    image.convertTo(image, CV_32F);
    // normalize just to visual better.
    normalize(image, image, 0, 1, NORM_MINMAX);
    imshow("image", image);

    Mat imgPSD;
    calcPSD(image, imgPSD);
    fftshift(imgPSD, imgPSD);
    normalize(imgPSD, imgPSD, 0, 1, NORM_MINMAX);
    imshow("psd", imgPSD);
    cout << "DFT" << endl;
    waitKey();

    cout << "IDFT" << endl;
    Mat H = Mat(image.size(), CV_32F, Scalar(1));
    filter2DFreq_vis(image, image_dst, H);

    int D0, n;
    cout << "Please input ideal low pass filter's D0:";
    cin >> D0;
    apply_ideal_filter(image, image_dst, D0, true);
    cout << "Please input ideal high pass filter's D0:";
    cin >> D0;
    apply_ideal_filter(image, image_dst, D0, false);

    cout << "Please input butterworth low pass filter's D0:";
    cin >> D0;
    cout << "Please input butterworth low pass filter's n:";
    cin >> n;
    apply_butterworth_filter(image, image_dst, D0, n, true);
    cout << "Please input butterworth high pass filter's D0:";
    cin >> D0;
    cout << "Please input butterworth high pass filter's n:";
    cin >> n;
    apply_butterworth_filter(image, image_dst, D0, n, false);

    return 0;
}
