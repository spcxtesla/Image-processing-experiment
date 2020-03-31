#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <numeric>
#include <iostream>
#include <vector>
#include <map>
#include <functional>
#include <string>

using namespace cv;
using namespace std;
using namespace std::placeholders;

float gauss_mean = 0, gauss_stdev = 15, pulse_prcnt = 0.01;

// line's point's x-aixs gap is depend on point number and image's width.
void plot_poly(Mat &imgIO, Mat &pts, const Scalar color, bool reset_scale=true)
{
    static float scale = 1.;
    Mat points;
    pts.convertTo(points, CV_32F);
    int width = imgIO.cols, height = imgIO.rows, pts_num = points.total();

    if (reset_scale) {
        double max_val;
        minMaxLoc(points, NULL, &max_val);
        scale = (double)height / max_val;
    }
    Mat tmpM(imgIO.size(), imgIO.type(), Scalar::all(0));

    for (int i = 1; i < pts_num; ++i)
        //Round operator should be done at last.
        line(tmpM,
            Point(cvRound((float)(i-1) / pts_num * width), height - cvRound(points.at<float>(i-1) * scale)),
            Point(cvRound((float)(i) / pts_num * width), height - cvRound(points.at<float>(i) * scale)),
            color, 2, LINE_AA, 0);

    imgIO += tmpM;
}

void add_gaussian_noise(const Mat & imgI, Mat & imgO, float mean, float stdev)
{
    cout << "Add gussian noise.\tmean:" << mean << "\tstdev:" << stdev << endl;
    Mat noise(imgI.size(), imgI.type());
    randn(noise, Scalar::all(mean), Scalar::all(stdev));
    add(imgI, noise, imgO);
}


void add_pulse_noise(const Mat &imgI, Mat &imgO, float percent, uchar value)
{
    CV_Assert(imgI.type() == CV_8U);

    cout << "Add pulse noise.\tpulse value:" << (int)value << endl;
    RNG rng(getTickCount());

    imgO = imgI.clone();
    int rows = imgI.rows, cols = imgI.cols;
    int counts = rows * cols * percent;
    for (int i = 0; i < counts; ++i)
        imgO.at<uchar>(rng.uniform(0, rows), rng.uniform(0, cols)) = value;
}

void filter2Dfun(const Mat &imgI, Mat &imgO, function<uchar(const Mat&)> pf, int m=3, int n=0)
{
    if (!n) n = m;
    cout << "\n\tsize(mxn):" << m << "x" << n;

    CV_Assert(m % 2 == 1);
    CV_Assert(n % 2 == 1);

    imgO = imgI.clone();
    // imgI.convertTo(imgO, CV_64F);

    int cols = imgI.cols, rows = imgI.rows;
    // top == (n - 1) / 2 == n / 2;  left == (m - 1) / 2 == m / 2;
    int top = n / 2, bottom = rows - top, left = m / 2, right = cols - left;
    for (int y = top; y < bottom; ++y) {
        for (int x = left; x < right; ++x) {
            Mat roi(imgI, Rect(x - left, y - top, m, n));
            imgO.at<uchar>(Point(x, y)) = pf(roi);
        }
    }
}

void apply_arithmetic_mean_filter(const Mat &imgI, Mat &imgO, int m=3, int n=0)
{
    if (!n) n = m;
    cout << "apply arithmetic mean filter.";
    auto fun = [=](const Mat &roi) -> uchar { return mean(roi)[0]; };
    filter2Dfun(imgI, imgO, fun, m, n);
    cout << endl;
}

void apply_geometric_mean_filter(const Mat &imgI, Mat &imgO, int m=3, int n=0)
{
    if (!n) n = m;
    cout << "apply geometric mean filter.";
    auto fun = [=](const Mat &roi) -> uchar {
        Mat tmpM;
        roi.convertTo(tmpM, CV_64F);
        // cv::pow(tmpM, (double)1/(m*n), tmpM);

// TODO: split 2-D into 1-D's may be better
        auto product = std::accumulate(tmpM.begin<double>(), tmpM.end<double>(), (double)1., [](double lhs, double rhs) -> double {
            if (!lhs) lhs = 1;
            if (!rhs) rhs = 1;
            return lhs * rhs;});
        product = pow(product, 1./m/n);
        // auto product = std::accumulate(tmpM.begin<double>(), tmpM.end<double>(), (double)1., multiplies<double>());
        return saturate_cast<uchar>(product);
    };
    filter2Dfun(imgI, imgO, fun, m, n);
    cout << endl;
}

void apply_harmonic_mean_filter(const Mat &imgI, Mat &imgO, int m=3, int n=0)
{
    if (!n) n = m;
    cout << "apply harmonic mean filter.";
    auto fun = [=](const Mat &roi) -> uchar {
        Mat tmpM;
        roi.convertTo(tmpM, CV_64F);
        divide(1, tmpM, tmpM);
        return saturate_cast<uchar>((double)m * n / sum(tmpM)[0]);
    };
    filter2Dfun(imgI, imgO, fun, m, n);
    cout << endl;
}

void apply_inverse_harmonic_mean_filter(const Mat &imgI, Mat &imgO, int Q, int m=3, int n=0)
{
    if (!n) n = m;
    cout << "apply inverse harmonic mean filter.\tQ:" << Q;
    auto fun = [=](const Mat &roi) -> uchar {
        Mat tmpM, tmpM_Qp1, tmpM_Q;
        roi.convertTo(tmpM, CV_64F);
        pow(tmpM, Q, tmpM_Q);
        pow(tmpM, Q+1, tmpM_Qp1);
        return saturate_cast<uchar>(sum(tmpM_Qp1)[0] / sum(tmpM_Q)[0]);
    };
    filter2Dfun(imgI, imgO, fun, m, n);
    cout << endl;
}

void apply_median_filter(const Mat &imgI, Mat &imgO, int m=3, int n=0)
{
    if (!n) n = m;
    cout << "apply median filter.";
    auto fun = [=](const Mat &roi) -> uchar {
        Mat tmpM = roi.clone();
        nth_element(tmpM.begin<uchar>(), tmpM.begin<uchar>() + tmpM.total()/2, tmpM.end<uchar>());
        return tmpM.at<uchar>(tmpM.total()/2);

    };
    filter2Dfun(imgI, imgO, fun, m, n);
    cout << endl;
}


void apply_adaptive_mean_filtering(const Mat &imgI, Mat &imgO, int m=3, int n=0)
{
    if (!n) n = m;
    cout << "apply adaptive mean filter.";
    float var_g = gauss_stdev;
    auto fun = [=](const Mat &roi) -> uchar {
        Mat mean_l, stdev_l;
        meanStdDev(roi, mean_l, stdev_l);
        double var_l = stdev_l.at<double>(0) * stdev_l.at<double>(0);
        if (var_g > var_l) {
            return saturate_cast<uchar>(mean_l.at<double>(0));
        } else {
            uchar val = roi.at<uchar>(roi.total()/2);
            return saturate_cast<uchar>(val - var_g / var_l * (val - mean_l.at<double>(0)));
        }
    };
    filter2Dfun(imgI, imgO, fun, m, n);
    cout << endl;
}

void apply_adaptive_median_filtering(const Mat &imgI, Mat &imgO, int S_max=9)
{
    int m = 3, r_max = (sqrt(S_max)-1)/2;
    cout << "apply adaptive median filter.";
    auto fun = [=](const Mat &roi0) -> uchar {
        uchar z_med, z_min, z_max, z_xy = roi0.at<uchar>(roi0.total()/2);
        Size sz;
        Point ofs;
        roi0.locateROI(sz, ofs);
        int w = sz.width, h = sz.height, cx = ofs.x + roi0.cols/2, cy = ofs.y + roi0.rows/2;
        Mat roi = roi0;

        // cout << endl << "cx:" << cx << "\tcy:" << cy << endl;

        for (int r = 1; r<=cx && r<=cy && r<w-cx && r<h-cy && r<=r_max; ++r) {
            auto z_minmax = minmax_element(roi.begin<uchar>(),roi.end<uchar>());
            z_min = *z_minmax.first, z_max = *z_minmax.second;

            Mat tmpM = roi.clone();
            nth_element(tmpM.begin<uchar>(), tmpM.begin<uchar>() + tmpM.total()/2, tmpM.end<uchar>());
            z_med =  tmpM.at<uchar>(tmpM.total()/2);

            if (z_min < z_med && z_med < z_max) {

                // cout << "return form program B." << endl;

                return (z_min < z_xy  && z_xy < z_max) ? z_xy : z_med;
            } else {
                roi.adjustROI(1, 1, 1, 1);

                // cout << "roi.size():" << roi.size() << endl;

            }
        }

        // cout << "size can't be bigger.return from program A." << endl;

        return z_med;
    };
    filter2Dfun(imgI, imgO, fun, m);
    cout << "~" << r_max *2 + 1 << "x" << r_max * 2 + 1 << endl;
}

void update_display(vector<string> wndwnms, vector<Mat*> images, string hist_wn="hist_img")
{
    namedWindow(hist_wn);
    Mat hist_img(images[0]->rows, images[0]->cols, CV_8UC3, Scalar::all(0));

    // origin image pdf, noise image pdf, processed image pdf
    vector<Mat> pdfs(3);
    float range[] = {0, 256};
    const float * hist_range = {range};
    int hist_size = 256;

    // cout << images[1] << endl;
    for (int i = 0; i < 3; ++i) {
        calcHist(images[i], 1, 0, Mat(), pdfs[i], 1, &hist_size, &hist_range);
        // plot_poly(hist_img,pdfs[i],Scalar(255*(0==i),255*(1==i),255*(2==i)));
        plot_poly(hist_img, pdfs[i], Scalar(255*(0==i),255*(1==i),255*(2==i)), i==0);
       
        imshow(wndwnms[i], *images[i]);
    }

    imshow(hist_wn, hist_img);
    waitKey();
    destroyWindow(hist_wn);
}

int main(int argc, char const *argv[])
{
    const char *image_path = (argc > 1) ? argv[1] : "img_test.jpg";

    Mat img, image, image_color, image_gray, image_noise, image_dst;
    img = imread(image_path, IMREAD_COLOR);
    if (img.empty()) {
        cout << "Cannot read image: " << image_path << std::endl;
        return -1;
    }

    image_color = img;
    cvtColor(image_color, image_gray, CV_BGR2GRAY);
    image = image_gray;

    vector<Mat*> imgs {&image, &image_noise, &image_dst};
    vector<string> wndnms {"image_src", "image_noise", "image_processed"};
    for (auto it : wndnms)
        namedWindow(it, CV_WINDOW_AUTOSIZE);

    vector<function<void(const Mat&, Mat&)>> add_noise_funs {
        bind(add_gaussian_noise, _1, _2, gauss_mean, gauss_stdev), 
        bind(add_pulse_noise, _1, _2, pulse_prcnt, 0), 
        bind(add_pulse_noise, _1, _2, pulse_prcnt, 255), 
        // TODO: [add_pulse_noise]
        [&](const Mat& imgI, Mat& imgO) -> void {
            add_pulse_noise(imgI, imgO, pulse_prcnt, 0); 
            add_pulse_noise(imgO, imgO, pulse_prcnt, 255);}
    };
    
    vector<function<void(const Mat&, Mat&)>> apply_mean_filters {
        bind(apply_arithmetic_mean_filter, _1, _2, 5, 5),
        bind(apply_geometric_mean_filter, _1, _2, 5, 5),
        bind(apply_harmonic_mean_filter, _1, _2, 5, 5),
        bind(apply_inverse_harmonic_mean_filter, _1, _2, 1, 5, 5)
    };


    for (auto apply_mean_filter : apply_mean_filters) {
        for (auto add_noise : add_noise_funs) {
            add_noise(image, image_noise);
            apply_mean_filter(image_noise, image_dst);
            update_display(wndnms, imgs);
            cout << endl;
        }
    }


    for (size_t i = 1; i < add_noise_funs.size(); ++i) {
        add_noise_funs[i](image, image_noise);

        apply_median_filter(image_noise, image_dst, 5);
        update_display(wndnms, imgs);
        cout << endl;

        apply_median_filter(image_noise, image_dst, 9);
        update_display(wndnms, imgs);
        cout << endl;
    }


    add_gaussian_noise(image, image_noise, gauss_mean, gauss_stdev);
    apply_arithmetic_mean_filter(image_noise, image_dst);
    update_display(wndnms, imgs);
    apply_adaptive_mean_filtering(image_noise, image_dst, 7);
    update_display(wndnms, imgs);
    cout << endl;


    add_pulse_noise(image, image_noise, pulse_prcnt, 255);
    add_pulse_noise(image_noise, image_noise, pulse_prcnt, 0);

    apply_median_filter(image_noise, image_dst, 7);
    update_display(wndnms, imgs);

    apply_adaptive_median_filtering(image_noise, image_dst, 7*7);
    update_display(wndnms, imgs);
    cout << endl;


    image = image_color;
    vector<Mat> bgr_planes;
    split(image, bgr_planes);
    for (auto &plane : bgr_planes) {
        add_gaussian_noise(plane, plane, gauss_mean, gauss_stdev);
        add_pulse_noise(plane, plane, pulse_prcnt, 0);
        add_pulse_noise(plane, plane, pulse_prcnt, 255);
    }
    merge(bgr_planes, image_noise);

    split(image_noise, bgr_planes);
    for(auto &plane : bgr_planes) {
        apply_arithmetic_mean_filter(plane, plane, 5);
    }
    merge(bgr_planes, image_dst);
    for (size_t i = 0; i < imgs.size(); ++i)
        imshow(wndnms[i], *imgs[i]);
    waitKey();

    split(image_noise, bgr_planes);
    for(auto &plane : bgr_planes) {
        apply_geometric_mean_filter(plane, plane, 5);
    }
    merge(bgr_planes, image_dst);
    for (size_t i = 0; i < imgs.size(); ++i)
        imshow(wndnms[i], *imgs[i]);
    waitKey();


    return 0;
}
