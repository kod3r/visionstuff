#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/photo/photo.hpp"
#include <stdio.h>
#include <vector>
#include <stdio.h>
//#include "histo.cpp"

using namespace std;
using namespace cv;

int histPick(Mat Oirgdisp, int &lower, int &upper);
int histRange(int pxv[], int index, int& lower, int& upper);
int segmentForeground(Mat &foreground, Mat &background);

int main(int argc, char* argv[])
{
    Mat disp, pMask, nMask, dMask, dnMask, valMat;
    int val;
    val = 116;
    disp =  imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    pMask = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    nMask = imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);

    dMask = nMask - pMask;
    dnMask = Scalar(255) - dMask;

    valMat = val * Mat::ones(disp.size(), CV_8U);
    bitwise_and(valMat, dMask, dMask);
    bitwise_and(disp, dnMask, dnMask);

    Mat newDisp;

    add(dMask, dnMask, newDisp);

    dMask = pMask - nMask;
    inpaint(newDisp, dMask, newDisp, 5, INPAINT_TELEA);
    imshow("old disp", disp);
    imshow("new disp", newDisp);

    waitKey(0);

    int lower, upper;
    int dispval;

    dispval = histPick(newDisp, lower, upper);
    Mat fg, bg;
    inRange(newDisp, Scalar(lower), Scalar(upper), fg);
    segmentForeground(fg, bg);
    Mat img;
    img = imread("65.jpg");
    resize(img, img, Size(img.cols/2, img.rows/2));
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    bitwise_and(img1, fg, fg);
    bitwise_and(img1, bg, bg);
    imshow("foreground", fg);
    imshow("background", bg);
    waitKey(0);

    return(0);
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

    for(int i=230; i>150; i--)
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
        maxindex = 150;
        sumPix = 0;

        for(int i=150; i>100; i--)
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