#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/photo/photo.hpp"
#include <stdio.h>
#include <vector>
#include <stdio.h>

using namespace std;
using namespace cv;

int getDisp(Mat img1, Mat img2, Mat& disp);

int main(int argc, char* argv[])
{
    Mat img1, img2, dispC, dispG, g1, g2;

    img1 = imread(argv[1]);
    img2 = imread(argv[2]);

    resize(img1, img1, Size(img1.cols/2, img1.rows/2));
    resize(img2, img2, Size(img2.cols/2, img2.rows/2));

    cvtColor(img1, g1, CV_BGR2Lab);
    cvtColor(img2, g2, CV_BGR2Lab);
    getDisp(img1, img2, dispC);
    getDisp(g1, g2, dispG);

    imshow("img1", img1);
    imshow("colordisp", dispC);
    imshow("labdisp", dispG);

    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    getDisp(g1, g2, dispG);
    imshow("graydisp", dispG);
    
    waitKey(0);

    imwrite("graydisp.jpg", dispG);
    return(0);
}

int getDisp(Mat img1, Mat img2, Mat& disp)
{
    Mat disp16;
    Mat disp8;
    StereoSGBM sbm;
    sbm.SADWindowSize = 7; // 5
    sbm.numberOfDisparities = 192;
    sbm.preFilterCap = 20;
    sbm.minDisparity = -64; // -64
    sbm.uniquenessRatio = 7; // 1
    sbm.speckleWindowSize = 180; //150
    sbm.speckleRange = 2;
    sbm.disp12MaxDiff = 20; // 10
    sbm.fullDP = false;
    sbm.P1 = 600;
    sbm.P2 = 2400;
    sbm(img1, img2, disp16);
    normalize(disp16, disp8, 0, 255, CV_MINMAX, CV_8U);

    Mat dispCrop(disp8, Rect(100, 100, disp8.cols - 200, disp8.rows - 200));
    Mat inpaintmask;
    threshold(dispCrop, inpaintmask, 10, 255, THRESH_BINARY_INV);
    imshow("inpaint mask", inpaintmask);
    inpaint(dispCrop, inpaintmask, disp, 10, INPAINT_NS);
    if (disp.cols > 0 && disp.rows > 0)
    {
        return 1;
    }
    return 0;
}