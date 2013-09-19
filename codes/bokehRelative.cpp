#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/photo/photo.hpp"
//#include <opencv/imgproc/imgproc.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

using namespace cv;
using namespace std;
Point point1;

int getDisparity(Mat g1, Mat g2, Mat &disp);
int getGaussianBlur(Mat img, Mat& retVal, int ksize);
int getThresh(Mat img, Mat& retVal, int l1, int l2, int h1, int h2);
int stackUp(vector<Mat>& layers, Mat& retVal);
int doMultiBlur(Mat img, Mat& retVal, Mat disp, Point p1);
int doCircBlur(Mat img, Mat& retVal, int radius);
int segmentForeground(Mat img, Mat &foreground);
int doGray(Mat img, Mat& retVal, Mat disp, Point p1);
int getRange(Mat disp, Point p1);

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
    img = imread(argv[1]);
    cvtColor(img, img, CV_RGBA2BGR);
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));
    //resize(img1, img1, Size(560, 630));
    //resize(img2, img2, Size(560, 630));
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    printf("%d %d\n", img1.cols, img1.rows);
    Mat newImg1(img1, Rect(70, 100, 500, 500));
    printf("newimg1\n");
    imshow("img", newImg1);
    cvSetMouseCallback("img", mouseHandler, NULL);
    getDisparity(g1, g2, disp);
    Mat newDisp;
    disp.copyTo(newDisp);
    //Mat newDisp(disp, Rect(70, 100, 500, 500));
    imshow("img", newImg1);
    waitKey(0);
    Mat blurImage;
    GaussianBlur(newImg1, blurImage, Size(19, 19), 15);
    doMultiBlur(newImg1, finImg, newDisp, point1);
    //doGray(newImg1, finImg, newDisp, point1);
    //addWeighted(newImg1, 0.3, finImg, 0.8, 0, finImg);
    imshow("final Image", finImg);
    waitKey(0);
    imwrite("multGaussBokeh.jpg", finImg);
    return 1;
}

int getDisparity(Mat g1, Mat g2, Mat &disp)
{
    Mat disp16;
    StereoSGBM sbm;
    sbm.SADWindowSize = 7; // 5
    sbm.numberOfDisparities = 128; //192
    sbm.preFilterCap = 4;
    sbm.minDisparity = -39; // -64
    sbm.uniquenessRatio = 9; // 1
    sbm.speckleWindowSize = 180; //150
    sbm.speckleRange = 2;
    sbm.disp12MaxDiff = 20; // 10
    sbm.fullDP = false;
    sbm.P1 = 600;
    sbm.P2 = 2400;
    sbm(g1, g2, disp16);
    normalize(disp16, disp, 0, 255, CV_MINMAX, CV_8U);
    Mat newDisp(disp, Rect(70, 100, 500, 500));
    Mat inpaintmask;
    imshow("prev disp", disp);
    threshold(newDisp, inpaintmask, 20, 255, THRESH_BINARY_INV);
    imshow("thresh paint", inpaintmask);
    inpaint(newDisp, inpaintmask, disp, 10, INPAINT_NS);
    //GaussianBlur(disp, disp, Size(5,5), 0);
    //Mat equalizeDisp;
    //medianBlur(disp, disp, 5);
    //equalizeHist(disp, equalizeDisp);
    //equalizeHist(disp, disp);

    imshow("disparity",disp);
    //imshow("disp16", disp16);
    //imshow("equalizeDisp", equalizeDisp);
    if (disp.cols > 0 && disp.rows > 0)
    {
        return 1;
    }
    return 0;
}

int doGray(Mat img, Mat& retVal, Mat disp, Point p1)
{
    int dispval, range;
    int l1, l2, h1, h2;
    dispval = disp.at<uchar>(p1.y, p1.x);
    range = getRange(disp, p1);
    //range = dispval/10;
    printf("%d %d\n", range, dispval);
    Mat thresh;
    l1 = dispval - range;
    l2 = dispval;
    h1 = dispval;
    h2 = dispval + range;

    inRange(disp, dispval - range, dispval + range, thresh);
    vector<Mat> threshvec;
    threshvec.push_back(thresh);
    threshvec.push_back(thresh);
    threshvec.push_back(thresh);
    merge(threshvec, thresh);
    //getThresh(disp, thresh, l1, l2, h1, h2);
    bitwise_and(img, thresh, retVal);
    //printf("first bitwise done\n");
    thresh = Scalar(255, 255, 255) - thresh;

    Mat gray;
    cvtColor(img, gray, CV_BGR2GRAY);
    //printf("%d\n", gray.channels());
    vector<Mat> grayvec;
    grayvec.push_back(gray);
    grayvec.push_back(gray);
    grayvec.push_back(gray);
    merge(grayvec, gray);
    //printf("%d\n", gray.channels());
    //imshow("thresh", thresh);
    //imshow("graey", gray);
    //waitKey(0);
    bitwise_and(gray, thresh, gray);
    //printf("second bitwise done\n");
    add(gray, retVal, retVal);

}

int getRange(Mat disp, Point p1)
{
    int dispval, range;
    dispval = disp.at<uchar>(p1.y, p1.x);
    range = dispval/6;
    Mat thresh;

    inRange(disp, dispval - range, dispval + range, thresh);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat drawing, kernel;
    int size=3;
    drawing = Mat::zeros(thresh.size(), CV_8UC3);
    Mat fg;
    //cvtColor(foreground, fg, CV_BGR2GRAY);
    findContours(thresh.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 1000)
        {
            printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }

    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));
    //erode(drawing, drawing, kernel, Point(-1, -1), 2);
    Mat temp;
    //dilate(drawing, temp, kernel, Point(-1, -1), 1);
    temp = drawing.clone();
    drawing = Mat::zeros(thresh.size(), CV_8UC1);
    contours.clear();
    hierarchy.clear();
    cvtColor(temp, temp, CV_BGR2GRAY);
    findContours(temp.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 2000)
        {
            printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    //dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    //foreground = drawing.clone();
    imshow("contours", drawing);
    //background = Scalar(255, 255, 255) - foreground;

    Mat dispClone;
    bitwise_and(disp, drawing, dispClone);
    imshow("dispClone", dispClone);
    Point minP, maxP;
    double minVal, maxVal;

    minMaxLoc(dispClone, &minVal, &maxVal, &minP, &maxP, thresh);
    printf("%lf %lf\n", minVal, maxVal);

    drawing = Mat::zeros(thresh.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    return (maxVal - dispval);
}

int doMultiBlur(Mat img, Mat& retVal, Mat disp, Point p1)
{
    int dispval, range, i, lval, hval;
    int l1, l2, h1, h2;
    vector<Mat> layers, blurs, finLayers;
    //Mat thresh, blur, bitwiseImg;
    int threshVal;

    dispval = disp.at<uchar>(p1.y, p1.x);
    range = getRange(disp, p1);
    range = dispval/12;
    printf("%d %d\n", range, dispval);
    lval = dispval+1;
    hval = dispval-1;
    for(i=1; i<4; i++)
    {
        l1 = lval - range;
        l2 = lval;
        h1 = hval;
        h2 = hval + range;
        printf("%d %d %d %d\n", l1, l2, h1, h2);
        Mat thresh;
        Mat seg;
        threshVal = getThresh(disp, thresh, l1, l2, h1, h2);
        if (!threshVal)
        {
            printf("break\n");
            break;
        }
        thresh.copyTo(seg);
        segmentForeground(thresh, seg);
        imshow("thresh", thresh);
        imshow("seg", seg);
        waitKey(0);
        layers.push_back(seg);

        lval = l1;
        hval = h2;
        range*=1.2;
    }
    Mat blurimg;
    //GaussianBlur(img, blurimg, Size(11, 11), 1);
    blurs.push_back(img);
    //blurs.push_back(img);
    for(i=1; i<layers.size(); i++)
    {
        Mat blur;
        GaussianBlur(img, blur, Size(2*i+1, 2*i+1), 2*i);
        //bilateralFilter(img, blur, 20, 100.0, 150.0, BORDER_DEFAULT);
        //doCircBlur(img, blur, 3*i);
        //imshow("blur", blur);
        //waitKey(0);
        blurs.push_back(blur);
    }
    int sigma = 2*i;
    int size = 2*i+1;
    Mat backLayer;
    backLayer = Mat::zeros(img.cols, img.rows, CV_8UC3);
    for(i=0; i<layers.size(); i++)
    {
        Mat bitwiseImg;
        bitwise_and(layers[i], blurs[i], bitwiseImg);
        add(backLayer, layers[i], backLayer);
        //imshow("thresh", layers[i]);
        //imshow("bitwiseImg", bitwiseImg);
        //waitKey(0);
        finLayers.push_back(bitwiseImg);
    }

    Mat blurImage;
    backLayer = Scalar(255, 255, 255) - backLayer;
    //imshow("backLayer", backLayer);
    //waitKey(0);
    GaussianBlur(img, blurImage, Size(size, size), sigma);
    //imshow("bluriamge", blurImage);
    bitwise_and(blurImage, backLayer, backLayer);
    //imshow("backLayer", backLayer);
    //waitKey(0);
    //finLayers.push_back(blurImage);
    finLayers.push_back(backLayer);
    //blurs.push_back(blurImage);

    stackUp(finLayers, retVal);

    return 1;


}

int getThresh(Mat img, Mat& retVal, int l1, int l2, int h1, int h2)
{
    Mat thresh1, thresh2, thresh;
    if (l2 < 0 && h1 > 255)
    {
        printf("return 0\n");
        return 0;
    }

    if (l1 < 0 && l2 < 0)
    {
        thresh1 = Mat::zeros(img.size(), CV_8U);
    }
    else if(l1 < 0)
    {
        l1 = 0;
        inRange(img, l1, l2-1, thresh1);
    }
    else
    {
        inRange(img, l1, l2-1, thresh1);
    }

    if (h2 > 255 && h1 > 255)
    {
        thresh2 = Mat::zeros(img.size(), CV_8U);
    }
    else if (h2 > 255)
    {
        h2 = 255;
        inRange(img, h1+1, h2, thresh2);
    }
    else
    {
        inRange(img, h1+1, h2, thresh2);
    }
    
    //imshow("thresh1", thresh1);
    //imshow("thresh2", thresh2);
    bitwise_or(thresh1, thresh2, thresh);

    vector<Mat> threshLayers;
    threshLayers.push_back(thresh);
    threshLayers.push_back(thresh);
    threshLayers.push_back(thresh);

    merge(threshLayers, retVal);

    if (retVal.size() == img.size())
    {
        return 1;
    }
    return 0;
}

int getGaussianBlur(Mat img, Mat& retVal, int ksize)
{
    GaussianBlur(img, retVal, Size(ksize, ksize), 0);
    if (retVal.size() == img.size())
    {
        return 1;
    }
    return 0;
}

int stackUp(vector<Mat>& layers, Mat& retVal)
{
    int i;
    retVal = Mat::zeros(layers[i].size(), CV_8UC3);
    //addWeighted(layers[0], 0.72, layers[1], 0.72, 0, retVal);
    for(i=0; i<layers.size(); i++)
    {
        add(retVal, layers[i], retVal);
    }
    if (retVal.size() == layers[0].size())
    {
        return 1;
    }
    return 0;
}

int doCircBlur(Mat img, Mat& retVal, int radius)
{
    Mat circ;
    int tSum;

    circ = Mat::zeros(31, 31, CV_64F);
    circle(circ, Point(15, 15), radius, (1, 1, 1), -1);
    tSum = sum(circ)[0];
    circ = circ/tSum;

    filter2D(img, retVal, -1, circ);
    if (retVal.size() == img.size())
    {
        return 1;
    }
    return 0;

}

int segmentForeground(Mat img, Mat &foreground)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat drawing, kernel;
    int size=3;
    drawing = Mat::zeros(img.size(), CV_8UC3);
    Mat fg;
    cvtColor(foreground, fg, CV_BGR2GRAY);
    findContours(fg.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 1000)
        {
            printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }

    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));
    //erode(drawing, drawing, kernel, Point(-1, -1), 2);
    Mat temp;
    //dilate(drawing, temp, kernel, Point(-1, -1), 1);
    temp = drawing.clone();
    drawing = Mat::zeros(img.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    cvtColor(temp, temp, CV_BGR2GRAY);
    findContours(temp.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 2000)
        {
            printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    //dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    foreground = drawing.clone();
    imshow("contours", drawing);
    //background = Scalar(255, 255, 255) - foreground;

    drawing = Mat::zeros(img.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    //bitwise_and(img, foreground, temp);
    //imwrite("checkblobs.jpg", temp);
    /*
    Mat grabcutimg;
    //doGrabCut(temp, grabcutimg, Point(360, 75), Point(450, 620));
    doGrabCut(temp, grabcutimg, Point(204, 96), Point(340, 542));
    //rectangle(grabcutimg, Point(204, 96), Point(340, 542), Scalar(255, 0, 0), 2);
    Mat threshGrabcut;
    threshold(grabcutimg, threshGrabcut, 180, 255, THRESH_BINARY);
    imshow("grabcut threshold", threshGrabcut);
    vector<Mat> grabcut3;
    grabcut3.push_back(threshGrabcut);
    grabcut3.push_back(threshGrabcut);
    grabcut3.push_back(threshGrabcut);
    merge(grabcut3, foreground);
    */
    //background = Scalar(255, 255, 255) - foreground;
    
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