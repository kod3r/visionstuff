#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/photo/photo.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

int getDisp(Mat g1, Mat g2, Mat &disp);
int doGrabCut(Mat img, Mat& retVal, Mat fg);
int histPick(Mat Oirgdisp, int &lower, int &upper);
int histRange(int pxv[], int index, int& lower, int& upper);
int segmentForeground(Mat &foreground, Mat &background);

int main(int argc, char* argv[])
{
    Mat img, g1, g2;
    Mat CM1, CM2, D1, D2, E, F, R, T;
    Mat R1, R2, P1, P2, Q;

    img = imread(argv[1]);
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));

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

    printf("%d %d\n", CM1.rows, CM1.cols);
    printf("%d %d\n", D1.rows, D1.cols);
    printf("%d %d\n", R1.rows, R1.cols);
    printf("%d %d\n", P1.rows, P1.cols);

    initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_32FC1, map2x, map2y);

    printf("%d %d\n", map1x.rows, map1y.cols);

    remap(img1, img1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    remap(img2, img2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    imwrite("rectifyLeft.jpg", img1);
    imwrite("rectifyRight.jpg", img2);
    Mat disp;
    resize(img1, img1, Size(img1.cols/2, img1.rows/2));
    resize(img2, img2, Size(img2.cols/2, img2.rows/2));
    imwrite("rectifyresizeLeft.jpg", img1);
    imwrite("rectifyresizeRight.jpg", img2);
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    getDisp(g1, g2, disp);
    imwrite("rectifiedDisparity.jpg", disp);
    int dispval;
    int lower, upper;
    dispval = histPick(disp, lower, upper);
    printf("dispval = %d\n", dispval);
    printf("lower = %d, upper = %d\n", lower, upper);
    imshow("disp", disp);

    Mat fg, bg;
    inRange(disp, Scalar(lower), Scalar(upper), fg);
    segmentForeground(fg, bg);

    imshow("fg", fg);
    //waitKey(0);
    Mat gbimg;
    doGrabCut(img1, gbimg, fg);

    imwrite(argv[3], gbimg);
    bitwise_and(img1, fg, fg);
    imwrite(argv[4], fg);

    imshow("grabcut", gbimg);
    imshow("withoutgrabCutImage", fg);
    waitKey(0);
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
    //Mat inpaintmask;
    //imshow("prev disp", disp);
    //threshold(disp, inpaintmask, 10, 255, THRESH_BINARY_INV);
    //imshow("thresh paint", inpaintmask);
    //inpaint(disp, inpaintmask, disp, 10, INPAINT_NS);
    if (disp.cols > 0 && disp.rows > 0)
    {
        return 1;
    }
    return 0;
}

int histPick(Mat Oirgdisp, int &lower, int &upper)
{
    // TODO: Find peaks and set range

    Mat disp(Oirgdisp, Rect(150, 0, Oirgdisp.cols - 300, Oirgdisp.rows));
    imshow("cropped", disp);
    int pxv[256] = {0};
    printf("%d %d\n", disp.rows, disp.cols);
    for(int i=0; i<disp.rows; i++)
    {
        for(int j=0; j<disp.cols; j++)
        {
            pxv[disp.at<uchar>(i, j)] += 1;
        }
    }
    printf("histogram created\n");
    for(int i=0; i<256; i++)
    {
        printf("%d - %d\n", i, pxv[i]);
    }

    int maxval=0;
    int val;
    int maxindex = 255;
    int sumPix=0;
    int totalPix = disp.cols * disp.rows;

    for(int i=255; i>200; i--)
    {
        val = pxv[i];
        sumPix += val;
        if (val > maxval)
        {
            maxval = val;
            maxindex = i;
        }
    }
    printf("interim dispval = %d\n", maxindex);
    printf("pixVal = %d\n", pxv[maxindex]);
    printf("sumPix = %d totalPix = %d\n", sumPix, totalPix);

    if (pxv[maxindex] < sumPix/20 || sumPix < totalPix/10)
    {
        printf("not enough pixels to support this value. Increasing the range.\n");
        maxval=0;
        maxindex = 200;
        sumPix = 0;

        for(int i=200; i>100; i--)
        {
            val = pxv[i];
            sumPix += val;
            if (val > maxval)
            {
                maxval = val;
                maxindex = i;
            }
        }

        printf("interim dispval = %d\n", maxindex);
        printf("pixVal = %d\n", pxv[maxindex]);
        printf("sumPix = %d totalPix = %d\n", sumPix, totalPix);

        if (pxv[maxindex] < sumPix/20 || sumPix < totalPix/20)
        {
            printf("not enough pixels to support this value. Increasing the range.\n");
            maxval=0;
            maxindex = 100;
            sumPix = 0;

            for(int i=100; i>50; i--)
            {
                val = pxv[i];
                sumPix += val;
                if (val > maxval)
                {
                    maxval = val;
                    maxindex = i;
                }
            }

            printf("interim dispval = %d\n", maxindex);
            printf("pixVal = %d\n", pxv[maxindex]);
            printf("sumPix = %d totalPix = %d\n", sumPix, totalPix);
        }

    }
    histRange(pxv, maxindex, lower, upper);
    return (maxindex);
}

int histRange(int pxv[], int index, int& lower, int& upper)
{
    int pixCount = pxv[index];
    int threshval = pixCount/10;
    int tolerance=3;
    int tCount=0;

    upper = index;
    lower = index;
    for(int i=index; i<256; i++)
    {
        printf("val = %d, threshval = %d\n", pxv[i], threshval);
        if (pxv[i] < threshval)
        {
            tCount++;
        }
        else
        {
            //tCount = 0;
            upper = i;
        }

        if(tCount > tolerance)
        {
            break;
        }
    }
    tCount = 0;
    for(int i=index; i>0; i--)
    {
        printf("val = %d, threshval = %d\n", pxv[i], threshval);
        if (pxv[i] < threshval)
        {
            tCount++;
        }
        else
        {
            //tCount = 0;
            lower = i;
        }

        if(tCount > tolerance)
        {
            break;
        }
    }
    /*
    if (upper > index + 15)
    {
        upper = index + 15;
    }
    if (lower < index - 15)
    {
        lower = index - 15;
    }*/

    if (upper != lower && upper != index)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int segmentForeground(Mat &foreground, Mat &background)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat drawing, kernel;
    int size=3;
    drawing = Mat::zeros(foreground.size(), CV_8UC3);
    findContours(foreground.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 15000)
        {
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }

    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));
    //erode(drawing, drawing, kernel, Point(-1, -1), 2);
    Mat temp;
    drawing.copyTo(temp);
    //dilate(drawing, temp, kernel, Point(-1, -1), 1);
    drawing = Mat::zeros(foreground.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    cvtColor(temp, temp, CV_BGR2GRAY);
    findContours(temp.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 15000)
        {
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    //dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    foreground = drawing.clone();
    background = Scalar(255, 255, 255) - foreground;
    return 1;
}

int doGrabCut(Mat img, Mat& retVal, Mat fg)
{
    Mat fgg;
    cvtColor(fg, fgg, CV_BGR2GRAY);
    //bitwise_and(img, fg, img);
    Mat tmp1, tmp2, mask, lut;
    tmp1 = Mat::zeros(1, 13*5, CV_64F);
    tmp2 = Mat::zeros(1, 13*5, CV_64F);
    mask = Mat::zeros(img.rows, img.cols, CV_8U);
    lut = Mat::zeros(256, 1, CV_8U);

    int x1=fg.cols, x2=0, y1=fg.rows, y2=0;

    for(int i=0; i<fg.rows; i++)
    {
        for(int j=0; j<fg.cols; j++)
        {
            if (fgg.at<uchar>(i, j) == 255)
            {
                if (j > x2)
                {
                    x2 = j;
                }

                if (i > y2)
                {
                    y2 = i;
                }

                if(j < x1)
                {
                    x1 = j;
                }
                if (i < y1)
                {
                    y1 = i;
                }

            }
        }
    }

    printf("Point1 = (%d, %d), Point2 = (%d, %d)\n", x1, y1, x2, y2);
    Point p1, p2;
    p1 = Point(x1, y1);
    p2 = Point(x2, y2);
    printf("Point1 = (%d, %d), Point2 = (%d, %d)\n", p1.x, p1.y, p2.x, p2.y);

    Mat erodefgg, kernel;
    int size = 3;
    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));

    
    printf("%d %d %d\n", GC_BGD, GC_FGD, GC_PR_BGD);
    dilate(fgg, erodefgg, kernel, Point(-1, -1), 3);
    imshow("erode", erodefgg);
    mask = erodefgg/255;
    //mask *= GC_PR_FGD;
    Mat pMask;
    pMask = erodefgg - fgg;
    imshow("diff", pMask);
    pMask /= 255;
    //pMask *= GC_PR_BGD;

    mask += pMask;
    //grabCut(img, mask, Rect(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y), tmp1, tmp2, 1, GC_INIT_WITH_RECT);
    grabCut(img, mask, Rect(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y), tmp1, tmp2, 3, GC_INIT_WITH_RECT);
    //grabCut(img, pMask, Rect(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y), tmp1, tmp2, 1, GC_INIT_WITH_MASK);
    lut.at<uchar>(1, 0) = 255;
    lut.at<uchar>(2, 0) = 64;
    lut.at<uchar>(3, 0) = 192;

    LUT(mask, lut, retVal);
    Mat threshretVal;
    threshold(retVal, threshretVal, 180, 255, THRESH_BINARY);
    //rectangle(fg, p1, p2, Scalar(255, 0, 255), 2);
    //imshow("rect", fg);
    //imshow("grabCut", retVal);
    //imshow("threshold", threshretVal);

    vector<Mat> grabcut3;
    grabcut3.push_back(threshretVal);
    grabcut3.push_back(threshretVal);
    grabcut3.push_back(threshretVal);
    merge(grabcut3, threshretVal);

    bitwise_and(img, threshretVal, retVal);
    //imshow("grabCutImage", threshretVal);

    //bitwise_and(img, fg, fg);
    //imshow("withoutgrabCutImage", fg);

    //imwrite(argv[3], threshretVal);
    //imwrite(argv[4], fg);

    //waitKey(0);
    return 1;
}