#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/photo/photo.hpp"
#include <stdio.h>
#include <string.h>
using namespace cv;
using namespace std;
Point point1;
Point point2;
Point point3;

int getDisparity(Mat g1, Mat g2, Mat &disp);
int getThreshold(Mat img, Point p1, int range, Mat &foreground);
int segmentForeground(Mat img, Mat &foreground, Mat &background);
int getBlurMaskedImage(Mat img, Mat &foreground);
int getMaskedImage(Mat img, Mat &foreground);
int addFgBg(Mat foreground, Mat background, Mat &img);
int getBlurMaskedGrayImage(Mat img, Mat &foreground);
int getMaskedGrayImage(Mat img, Mat &foreground);
int filterDisp(Mat &img);
int stickImage(Mat &foreground, Mat &background);
int doGrabCut(Mat img, Mat& retVal, Point p1, Point p2);

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        /* left button clicked. ROI selection begins */
        if (point1.x == 0)
        {
            point1 = Point(x, y);
            printf("points1 %d %d\n", x, y);
        }
        else if (point2.x == 0)
        {
            point2 = Point(x, y);
            printf("points2 %d %d\n", x, y);
        }
        else if(point3.x == 0)
        {
            point3 = Point(x, y);
            printf("points3 %d %d\n", x, y);
        }
    }
}

int main(int argc, char* argv[])
{
    point1 = Point(0, 0); // pre defined
    point2 = Point(0, 0);
    point3 = Point(0, 0);
    Mat img, g1, g2, disp, foreground, background, finImg;
    int fg;
    img = imread(argv[1]);
    fg = atoi(argv[2]);
    cvtColor(img, img, CV_RGBA2BGR);
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));
    printf("%d %d\n", img2.cols, img2.rows);
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    imshow("img", img1);
    cvSetMouseCallback("img", mouseHandler, NULL);
    waitKey(0);

    getDisparity(g1, g2, disp);
    //filterDisp(disp); // never do this
    int dispval;
    dispval = disp.at<uchar>(point1.x, point1.y);
    int range = dispval/10;
    getThreshold(disp, point1, range, foreground);
    segmentForeground(img1, foreground, background);
    getMaskedImage(img1, foreground);
    imshow("foregroundprev", foreground);
    Mat eiffel;
    eiffel = imread("img_left4.jpg");
    stickImage(foreground, eiffel);
    imshow("foreground", foreground);
    imshow("background", eiffel);
    //imshow("finImg", finImg);
    //cvtColor(finImg, finImg, CV_BGR2GRAY);
    imwrite("eiffel-out.jpg", eiffel);
    waitKey(0);
    return 1;
}

int getDisparity(Mat g1, Mat g2, Mat &disp)
{
    Mat disp16;
    StereoSGBM sbm;
    sbm.SADWindowSize = 7; // 5
    sbm.numberOfDisparities = 192;
    sbm.preFilterCap = 4;
    sbm.minDisparity = -32; // -64
    sbm.uniquenessRatio = 9; // 1
    sbm.speckleWindowSize = 180; //150
    sbm.speckleRange = 2;
    sbm.disp12MaxDiff = 20; // 10
    sbm.fullDP = false;
    sbm.P1 = 600;
    sbm.P2 = 2400;
    /*
    sbm.SADWindowSize = 5; // 5
    sbm.numberOfDisparities = 112;
    sbm.preFilterCap = 61;
    sbm.minDisparity = -39; // -64
    sbm.uniquenessRatio = 1; // 1
    sbm.speckleWindowSize = 180; //150
    sbm.speckleRange = 2;
    sbm.disp12MaxDiff = 20; // 10
    sbm.fullDP = false;
    sbm.P1 = 600;
    sbm.P2 = 2400;
    */
    sbm(g1, g2, disp16);
    normalize(disp16, disp, 0, 255, CV_MINMAX, CV_8U);
    imshow("disparity",disp);
    Mat inpaintmask;
    imshow("prev disp", disp);
    threshold(disp, inpaintmask, 10, 255, THRESH_BINARY_INV);
    imshow("thresh paint", inpaintmask);
    inpaint(disp, inpaintmask, disp, 10, INPAINT_NS);
    imshow("disparity",disp);
    if (disp.cols > 0 && disp.rows > 0)
    {
        return 1;
    }
    return 0;
}

int getThreshold(Mat img, Point p1, int range, Mat &foreground)
{
    int disval;
    disval = img.at<uchar>(p1.y, p1.x);
    inRange(img, disval - range, disval + range, foreground);
    medianBlur(foreground, foreground, 9);
    return 1;
}

int segmentForeground(Mat img, Mat &foreground, Mat &background)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat drawing, kernel;
    int size=3;
    drawing = Mat::zeros(img.size(), CV_8UC3);
    findContours(foreground.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 15000)
        {
            printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }

    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));
    erode(drawing, drawing, kernel, Point(-1, -1), 2);
    Mat temp;
    dilate(drawing, temp, kernel, Point(-1, -1), 1);
    drawing = Mat::zeros(img.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    cvtColor(temp, temp, CV_BGR2GRAY);
    findContours(temp.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 15000)
        {
            printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    //dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    foreground = drawing.clone();
    imshow("contours", drawing);
    background = Scalar(255, 255, 255) - foreground;

    drawing = Mat::zeros(img.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    bitwise_and(img, foreground, temp);
    imwrite("checkblobs.jpg", temp);
    
    Mat grabcutimg;
    //doGrabCut(temp, grabcutimg, Point(360, 75), Point(450, 620));
    //doGrabCut(temp, grabcutimg, Point(187, 55), Point(340, 606));
    doGrabCut(temp, grabcutimg, point2, point3);
    //rectangle(grabcutimg, Point(204, 96), Point(340, 542), Scalar(255, 0, 0), 2);
    Mat threshGrabcut;
    threshold(grabcutimg, threshGrabcut, 180, 255, THRESH_BINARY);
    imshow("grabcut threshold", threshGrabcut);
    vector<Mat> grabcut3;
    grabcut3.push_back(threshGrabcut);
    grabcut3.push_back(threshGrabcut);
    grabcut3.push_back(threshGrabcut);
    merge(grabcut3, foreground);
    background = Scalar(255, 255, 255) - foreground;
    
    /*
    cvtColor(temp, temp, CV_BGR2GRAY);
    Canny(temp, temp, 100, 200);
    imshow("temp", temp);
    findContours(temp.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    printf("number of contours = %ld\n", contours.size());
    for (int i=0; i<contours.size(); i++)
    {
        drawContours(drawing, contours, i, Scalar(255, 0, 255), 1, 8, hierarchy, 0, Point());
        //imshow("contours in the making", drawing);
        //waitKey(0);
    }
    imshow("contours on color", drawing);
    */
    return 1;
}

int doGrabCut(Mat img, Mat& retVal, Point p1, Point p2)
{
    Mat tmp1, tmp2, mask, lut;
    tmp1 = Mat::zeros(1, 13*5, CV_64F);
    tmp2 = Mat::zeros(1, 13*5, CV_64F);
    mask = Mat::zeros(img.rows, img.cols, CV_8U);
    lut = Mat::zeros(256, 1, CV_8U);

    grabCut(img, mask, Rect(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y), tmp1, tmp2, 3, GC_INIT_WITH_RECT);
    lut.at<uchar>(1, 0) = 255;
    lut.at<uchar>(2, 0) = 64;
    lut.at<uchar>(3, 0) = 192;

    LUT(mask, lut, retVal);
    imshow("grabcut", retVal);
    return 1;
}

int getBlurMaskedGrayImage(Mat img, Mat &foreground)
{
    Mat blur, blurGray;
    cvtColor(img, blurGray, CV_BGR2GRAY);
    GaussianBlur(blurGray, blurGray, Size(9, 9), 11, 11);
    vector<Mat> gray;
    gray.push_back(blurGray);
    gray.push_back(blurGray);
    gray.push_back(blurGray);
    merge(gray, blur);
    bitwise_and(blur, foreground, foreground);
    //bilateralFilter(foreground, newImg, 20, 100.0, 150.0, BORDER_DEFAULT);
    //GaussianBlur(newImg, foreground, Size(9, 9), 11, 11);
    //foreground = newImg.clone();
    return 1;
}
int getBlurMaskedImage(Mat img, Mat &foreground)
{
    Mat blur;
    GaussianBlur(img, blur, Size(9, 9), 11, 11);
    bitwise_and(blur, foreground, foreground);
    return 1;
}
int getMaskedImage(Mat img, Mat &foreground)
{
    bitwise_and(img, foreground, foreground);
    return 1;
}

int getMaskedGrayImage(Mat img, Mat &foreground)
{
    Mat g, bitwise;
    cvtColor(img, g, CV_BGR2GRAY);
    vector<Mat> gray;
    gray.push_back(g);
    gray.push_back(g);
    gray.push_back(g);
    merge(gray, bitwise);
    bitwise_and(bitwise, foreground, foreground);
    return 1;
}

int addFgBg(Mat foreground, Mat background, Mat &img)
{
    Mat tempImg;
    add(foreground, background, img);
    return 1;
}

int filterDisp(Mat &img)
{
    // This filter will ruin it
    Mat filter = Mat::ones(5,5, CV_32FC1)/25.0;
    filter.at<float>(1, 0) = 2/25.0;
    filter.at<float>(0, 1) = 2/25.0;
    filter.at<float>(3, 0) = 2/25.0;
    filter.at<float>(0, 3) = 2/25.0;
    filter.at<float>(2, 0) = 3/25.0;
    filter.at<float>(0, 2) = 3/25.0;
    filter.at<float>(2, 4) = 3/25.0;
    filter.at<float>(4, 2) = 3/25.0;
    filter.at<float>(1, 1) = 4/25.0;
    filter.at<float>(1, 3) = 4/25.0;
    filter.at<float>(3, 1) = 4/25.0;
    filter.at<float>(3, 3) = 4/25.0;
    filter.at<float>(2, 1) = 6/25.0;
    filter.at<float>(1, 2) = 6/25.0;
    filter.at<float>(2, 3) = 6/25.0;
    filter.at<float>(3, 2) = 6/25.0;
    filter.at<float>(2, 2) = 9/25.0;

    filter2D(img, img, CV_8U, filter);
}

int stickImage(Mat &foreground, Mat &background)
{
    Mat bitwise;
    int i, j;
    resize(background, background, Size(foreground.cols, foreground.rows));
    for(i=0; i<background.rows; i++)
    {
        for(j=0; j<background.cols; j++)
        {
            if (foreground.at<Vec3b>(i,j)[0] != 0)
            {
                background.at<Vec3b>(i,j) = foreground.at<Vec3b>(i,j);
            }
        }
    }

}