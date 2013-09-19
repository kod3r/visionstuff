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
    point1 = Point(0, 0);
    Mat img, g1, g2;
    Mat disp, disp8;
    img = imread(argv[1]);
    cvtColor(img, img, CV_RGBA2BGR);
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));
    printf("%d %d\n", img2.cols, img2.rows);
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    
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
    sbm(g1, g2, disp);

    
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

    imshow("left", img1);
    imshow("right", img2);
    imshow("disp", disp8);
    //imwrite("disp.jpg", disp8);
    cvSetMouseCallback("left", mouseHandler, NULL);
    int i, j, sum = 0, count = 0, val;
    for(i = 0; i < disp8.cols; i++)
    {
        for(j = 0 ; j < disp8.rows; j++)
        {
            val = disp8.at<uchar>(j, i);
            if (val != 0)
            {
                    sum += val;    
                    count++;
            }
            
        }
    }
        
    int ave;
    ave = sum/count;
    printf("%d %d %d %d\n", img1.cols, img1.rows, disp8.rows, disp8.cols);
    Mat img_thresh;
    printf("%d\n", ave);
    threshold(disp8, img_thresh, ave+10, 255,THRESH_BINARY);
    medianBlur(img_thresh, img_thresh, 9);
    Mat kernel;
    int size=3;
    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));
    //erode(img_thresh, img_thresh, kernel, Point(-1, -1), 3);
    //dilate(img_thresh, img_thresh, kernel, Point(-1, -1), 3);
    Mat bitwise, canny_img, drawing;
    Canny(img_thresh, canny_img, 100, 200);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    waitKey(0);
    
    short disval;
    disval = disp.at<short>(point1.y, point1.x);
    printf("disp = %d %d %d\n",disval, point1.x, point1.y);
    Mat img_t, img_t1;
    int disintval;
    disintval = disp8.at<uchar>(point1.y, point1.x);
    printf("disp = %d %d %d\n",disintval, point1.x, point1.y);
    inRange(disp8, disintval-8, disintval+8, img_t1);
    
    inRange(disp, disval-15, disval+15, img_t);
    medianBlur(img_t1, img_t1, 9);
    //erode(img_t1, img_t1, kernel, Point(-1, -1), 3);
    //dilate(img_t1, img_t1, kernel, Point(-1, -1), 3);

    contours.clear();
    drawing = Mat::zeros(canny_img.size(), CV_8UC3);
    findContours(img_t1.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 15000)
        {
            printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    imshow("drawing1", drawing);
    erode(drawing, drawing, kernel, Point(-1, -1), 2);
    dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    img_t1 = drawing.clone();
    Mat eiffel;
    eiffel = imread("eiffel-tower-day.jpg");
    resize(eiffel, eiffel, Size(img1.cols, img1.rows));
    bitwise_and(img1, drawing, bitwise);
    Canny(bitwise, canny_img, 150, 200);
    for(i=0; i<eiffel.rows; i++)
    {
        for(j=0; j<eiffel.cols; j++)
        {
            if (img_t1.at<Vec3b>(i,j)[0] != 0)
            {
                eiffel.at<Vec3b>(i,j) = img1.at<Vec3b>(i,j);
            }
        }
    }
    imshow("canny", canny_img);
    imshow("new threshold", img_t);
    imshow("new threshold1", img_t1);
    imshow("new bitwise", bitwise);
    imshow("eiffel", eiffel);
    imwrite("eiffel-out.jpg", eiffel);
    waitKey(0);
    
    return(0);
}

