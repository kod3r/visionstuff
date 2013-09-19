#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <string.h>
using namespace cv;
using namespace std;
Point point1;

int getDisparity(Mat g1, Mat g2, Mat &disp);
int getThreshold(Mat img, Point p1, int range, Mat &foreground);
int segmentForeground(Mat img, Mat &foreground, Mat &background);
int getBlurMaskedImage(Mat img, Mat &foreground);
int getMaskedImage(Mat img, Mat &foreground);
int addFgBg(Mat foreground, Mat background, Mat &img);
int getAve(Mat &img, int x, int y, int size);
int getave(Mat &img, int x, int y, int size);
void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        printf("points %d %d\n", x, y);
    }
}

int main(int argc, char* argv[])
{
    point1 = Point(291, 352); // pre defined
    Mat img, g1, g2, disp, foreground, background, finImg;
    int fg;
    img = imread(argv[1]);
    fg = atoi(argv[2]);
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));
    printf("%d %d\n", img2.cols, img2.rows);
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    imshow("img", img1);
    cvSetMouseCallback("img", mouseHandler, NULL);
    waitKey(0);

    getDisparity(g1, g2, disp);
    getThreshold(disp, point1, 7, foreground);
    segmentForeground(img1, foreground, background);
    if (fg == 1)
    {
        getBlurMaskedImage(img1, background);
        getMaskedImage(img1, foreground);
        addFgBg(foreground, background, finImg);
        imwrite("backgroundblur.jpg", finImg);
    }
    else
    {
        getBlurMaskedImage(img1, foreground);
        getMaskedImage(img1, background);
        addFgBg(foreground, background, finImg);
        imwrite("foregorundblur.jpg", finImg);
    }
    imshow("foreground", foreground);
    imshow("background", background);
    imshow("finImg", finImg);
    waitKey(0);
    return 1;
}

int getDisparity(Mat g1, Mat g2, Mat &disp)
{
    Mat disp16;
    StereoSGBM sbm;

    sbm.SADWindowSize = 5;
    sbm.numberOfDisparities = 192;
    sbm.preFilterCap = 4;
    sbm.minDisparity = -64;
    sbm.uniquenessRatio = 1;
    sbm.speckleWindowSize = 150;
    sbm.speckleRange = 2;
    sbm.disp12MaxDiff = 10;
    sbm.fullDP = false;
    sbm.P1 = 600;
    sbm.P2 = 2400;

    sbm(g1, g2, disp16);
    normalize(disp16, disp, 0, 255, CV_MINMAX, CV_8U);
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
    dilate(drawing, drawing, kernel, Point(-1, -1), 1);

    foreground = drawing.clone();
    background = Scalar(255, 255, 255) - foreground;
    return 1;
}

int getBlurMaskedImage(Mat img, Mat &foreground)
{
    Mat blur;
    bitwise_and(img, foreground, foreground);
    //Mat newImg;
    GaussianBlur(img, blur, Size(9, 9), 11, 11);
    //bilateralFilter(foreground, newImg, 20, 100.0, 150.0, BORDER_DEFAULT);
    //GaussianBlur(newImg, foreground, Size(9, 9), 11, 11);
    //foreground = newImg.clone();
    return 1;
}

int getMaskedImage(Mat img, Mat &foreground)
{
    bitwise_and(img, foreground, foreground);
    return 1;
}

int addFgBg(Mat foreground, Mat background, Mat &img)
{
    Mat tempImg;
    add(foreground, background, img);
    int i, j, fval, bval;
    int count=0;
    for(i=30; i<foreground.rows-30; i++)
    {
        for(j=30; j<foreground.cols-30; j++)
        {
            fval = foreground.at<Vec3b>(i,j)[0];
            bval = foreground.at<Vec3b>(i,j-1)[0];
            //printf("%d %d\n", fval, bval);
            if ((fval == 0 && bval != 0) || (fval != 0 && bval == 0))
            {
                //printf("%d %d\n",i,j);
                getave(img, j, i, 5);
                count++;
            }
        }
    }
    printf("print\n");
    printf("border points = %d\n", count);
    printf("%d %d\n", img.cols, img.rows);
    //addWeighted(foreground, 1.0, background, 0.9, 1.0, img);
    //img = tempImg.clone();
    //img = foreground + background;
    //GaussianBlur(img, img, Size(3,3),1,1);
    return 1;
}

int getAve(Mat &img, int x, int y, int size)
{
    int b, g, r, i, j;
    Scalar bgr;
    Mat subImg;
    subImg = img(Range(y-size/2, y+size/2+1), Range(x-size/2, x+size/2+1));
    bgr = sum(subImg);
    b = bgr[0]/(size*size);
    g = bgr[1]/(size*size);
    r = bgr[2]/(size*size);
    printf("%d %d %d\n",b,g,r);
    img.at<Vec3b>(i,j) = Vec3b(b, g, r);
    //printf("%d %d %d\n",b,g,r);
    return 1;
}
int getave(Mat &img, int x, int y, int size)
{
    int b, g, r, i, j;
    Scalar bgr;
    Mat subImg;
    Mat sqImg = Mat::ones(size+2, size+2, CV_8UC3);
    int sqImgx=0, sqImgy=0;
    for(i=y-size/2; i<y+size/2; i++)
    {
        for(j=x-size/2; j<x+size/2; j++)
        {
            subImg = img(Range(i-size/2, i+size/2+1), Range(j-size/2, j+size/2+1));
            bgr = sum(subImg);
            b = bgr[0]/(size*size);
            g = bgr[1]/(size*size);
            r = bgr[2]/(size*size);
            sqImg.at<Vec3b>(sqImgy, sqImgx) = Vec3b(b, g, r);
            sqImgx++;
        }
        sqImgy++;
    }
    sqImgy = sqImgx = 0;
    for(i=y-size/2; i<y+size/2; i++)
    {
        for(j=x-size/2; j<x+size/2; j++)
        {
            img.at<Vec3b>(i, j) = sqImg.at<Vec3b>(sqImgy, sqImgx);
            sqImgx++;
        }
        sqImgy++;
    }
    return 1;
}