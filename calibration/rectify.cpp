#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <dirent.h>

using namespace cv;
using namespace std;
int getDisp(Mat g1, Mat g2, Mat &disp);

int main(int argc, char* argv[])
{
    Mat img, g1, g2;
    Mat CM1, CM2, D1, D2, E, F, R, T;
    Mat R1, R2, P1, P2, Q;

    img = imread(argv[1]);
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));

    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);

    FileStorage fs1(argv[2], FileStorage::READ);

    fs1["CM1"] >> CM1;
    fs1["CM2"] >> CM2;
    fs1["D1"] >> D1;
    fs1["D2"] >> D2;
    fs1["E"] >> E;
    fs1["F"] >> F;
    fs1["R"] >> R;
    fs1["R1"] >> R1;
    fs1["R2"] >> R2;
    fs1["P1"] >> P1;
    fs1["P2"] >> P2;
    fs1["Q"] >> Q;

    fs1.release();

    Mat map1x, map1y, map2x, map2y;
    Mat imgU1, imgU2;

    initUndistortRectifyMap(CM1, D1, R1, P1, g1.size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(CM2, D2, R2, P2, g1.size(), CV_32FC1, map2x, map2y);

    remap(g1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    remap(g2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

    Mat disp;
    resize(imgU1, imgU1, Size(imgU1.cols/2, imgU1.rows/2));
    resize(imgU2, imgU2, Size(imgU2.cols/2, imgU2.rows/2));
    getDisp(imgU1, imgU2, disp);

    imwrite("LEFT.pgm", imgU1);
    imwrite("RIGHT.pgm", imgU2);

    remap(img1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    remap(img2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

    imwrite("LEFT.jpg", imgU1);
    imwrite("RIGHT.jpg", imgU2);

    imshow("disparity", disp);
    waitKey(0);
    imwrite(argv[3], disp);
    return(0);
}

int getDisp(Mat g1, Mat g2, Mat &disp)
{
    Mat disp16;
    StereoSGBM sbm;
    sbm.SADWindowSize = 7; // 5
    sbm.numberOfDisparities = 112;
    sbm.preFilterCap = 10;
    sbm.minDisparity = -64; // -64
    sbm.uniquenessRatio = 1; // 1
    sbm.speckleWindowSize = 150; //150
    sbm.speckleRange = 2;
    sbm.disp12MaxDiff = 20; // 10
    sbm.fullDP = false;
    sbm.P1 = 800;
    sbm.P2 = 3200;
    sbm(g1, g2, disp16);
    Mat tempDisp;
    normalize(disp16, tempDisp, 0, 255, CV_MINMAX, CV_8U);
    //medianBlur(disp, disp, 5);
    bilateralFilter(tempDisp, disp, 5, 50, 0);
    if (disp.cols > 0 && disp.rows > 0)
    {
        return 1;
    }
    return 0;
}