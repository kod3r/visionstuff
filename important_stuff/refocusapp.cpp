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
Point point1;
Point pointC;
bool pointCdrawfg;
bool pointCdrawbg;

int getDisp(Mat g1, Mat g2, Mat &disp);
int fillDisp(Mat& disp);
int getThreshold(Mat img, Point p1, int range, Mat &foreground);
int segmentForeground(Mat img, Mat &foreground, Mat &background, vector<vector<Point> >& contours);
int getBlurMaskedImage(Mat img, Mat &foreground);
int getMaskedImage(Mat img, Mat &foreground);
int addFgBg(Mat foreground, Mat background, Mat &img);
int getBlurMaskedGrayImage(Mat img, Mat &foreground);
int getMaskedGrayImage(Mat img, Mat &foreground);
int deFocus(Mat img, Mat& foreground, int size, int radius);
int deFocus8(Mat img, Mat& foreground, int size, int w, int h);
int doBokeh(Mat disp, Mat img);
int doBokehImg(Mat disp, Mat img, Mat& foreground);
int getSepia(Mat img, Mat &foreground);
int doBokehImgRelative(Mat disp, Mat img, Mat& foreground, Point p1);
int getGaussianBlur(Mat img, Mat& retVal, int ksize);
int getThresh(Mat img, Mat& retVal, int l1, int l2, int h1, int h2);
int stackUp(vector<Mat>& layers, Mat& retVal);
int doMultiBlur(Mat img, Mat& retVal, Mat disp, Point p1);
int doCircBlur(Mat img, Mat& retVal, int radius);
int getDisparity(Mat img, Mat &disp);
int segmentBlurs(Mat img, Mat &foreground);
int doOilPaint(Mat src, Mat& foreground);
int doGraySingle(Mat img, Mat& retVal, Mat disp, Point p1);
int getRange(Mat disp, Point p1);
int stickImage(Mat &foreground, Mat &background);
int histPick(Mat disp);
int getThresholdHist(Mat img, int dispval, int range, Mat &foreground);
int doMultiBlurHist(Mat img, Mat& retVal, Mat disp, int dispval);
int updateFgBg(Mat fgRaw, Mat bgRaw, bool updateFg, Mat& finImg, Mat img1, Mat disp);

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        printf("points %d %d\n", x, y);
    }
}

void toolBrush(int event, int x, int y, int flags, void* param)
{
    pointC = Point(x, y);
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        /* left button clicked. ROI selection begins */
        pointCdrawfg = !pointCdrawfg;
        pointCdrawbg = false;
    }
    if (event == CV_EVENT_RBUTTONDOWN)
    {
        /* left button clicked. ROI selection begins */
        pointCdrawbg = !pointCdrawbg;
        pointCdrawfg = false;
    }

}

int main(int argc, char* argv[])
{
    Mat img, foreground, background, disp, finImg;
    int currentMode;
    vector<vector<Point> > contours;
    
    img = imread(argv[1]);
    //cvtColor(img, img, CV_RGBA2BGR);
    resize(img, img, Size(img.cols/2, img.rows/2));
    currentMode = atoi(argv[2]);

    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    point1 = Point(0, 0); // to get from android
    imshow("img", img1);
    cvSetMouseCallback("img", mouseHandler, NULL);
    waitKey(0);
    getDisparity(img, disp);
    imshow("disp", disp);
    //fillDisp(disp);
    imshow("newDisp", disp);
    imwrite("refocus_inpaint_disp.jpg", disp);
    int dispval;
    dispval = histPick(disp);
    getThresholdHist(disp, dispval, 10, foreground);
    //dispval = getThreshold(disp, point1, 12, foreground);
    printf("dispval = %d\n", dispval);
    imshow("thresh foreground", foreground);
    waitKey(0);
    segmentForeground(img1, foreground, background, contours);

    Mat fgRaw, bgRaw;
    foreground.copyTo(fgRaw);
    background.copyTo(bgRaw);

    Mat layerAf, layerAb;
    cvtColor(foreground, layerAf, CV_BGR2GRAY);
    cvtColor(background, layerAb, CV_BGR2GRAY);
    imshow("seg foreground", foreground);
    imshow("seg background", background);
    //Mat meanshiftMat;
    //pyrMeanShiftFiltering(img1, meanshiftMat, 10, 10);
    //imshow("meanshiftMat", meanshiftMat);
    //waitKey(0);
    int tLen=0;
    for(int i=0; i<contours.size(); i++)
    {
        for(int j=0; j<contours[i].size(); j++)
        {
            tLen++;
        }
    }

    float cPoints[2*tLen];
    tLen=0;
    for(int i=0; i<contours.size(); i++)
    {
        for(int j=0; j<contours[i].size(); j++)
        {
            cPoints[tLen] = contours[i][j].x;
            tLen++;
            cPoints[tLen] = contours[i][j].y;
            tLen++;
        }
    }
    
    if(currentMode==1)
    {
        Mat blurBackground;
        doMultiBlur(img1, blurBackground, disp, point1);
        //doMultiBlurHist(img1, blurBackground, disp, dispval);
        bitwise_and(background, blurBackground, background);
        getMaskedImage(img1, foreground);
    }
    else if(currentMode==2)
    {
        doOilPaint(img1, background);
        getMaskedImage(img1, foreground);
    }
    else if(currentMode==3)
    {
        getMaskedGrayImage(img1, background);
        getMaskedImage(img1, foreground);
    }
    else if(currentMode==4)
    {
        getSepia(img1, background);
        getMaskedImage(img1, foreground);
    }
    else if(currentMode == 5)
    {
        Mat stickimg;
        stickimg = imread(argv[3]);
        resize(stickimg, stickimg, Size(background.cols, background.rows));
        bitwise_and(background, stickimg, stickimg);
        stickimg.copyTo(background);
        getMaskedImage(img1, foreground);
    }
    else if (currentMode == 6)
    {
        Mat stickimg;
        stickimg = imread(argv[3]);
        resize(stickimg, stickimg, Size(foreground.cols, foreground.rows));
        bitwise_and(background, stickimg, stickimg);
        stickimg.copyTo(background);
        getMaskedImage(img1, foreground);
    }
    

    cvtColor(foreground, foreground, CV_BGR2RGBA);
    cvtColor(background, background, CV_BGR2RGBA);
    
    vector<Mat> rgbam;
    split(foreground, rgbam);
    rgbam[3] = layerAf;
    merge(rgbam, foreground);
    rgbam.clear();
    
    split(background, rgbam);
    rgbam[3] = layerAb;
    merge(rgbam, background);
    rgbam.clear();

    imwrite("refocus_fg.png", foreground);
    imwrite("refocus_bg.png", background);
    cvtColor(foreground, foreground, CV_RGBA2BGR);
    cvtColor(background, background, CV_RGBA2BGR);
    addFgBg(foreground, background, finImg);
    imshow("final", finImg);
    imwrite("refocus_final.png", finImg);
    waitKey(0);
    return(0);
    /*
    Mat displayImage;
    Mat displayDisp;
    disp.copyTo(displayDisp);
    finImg.copyTo(displayImage);
    pointC = Point(0, 0);
    pointCdrawfg = false;
    pointCdrawbg = false;
    int rad = 10;
    int k;

    imshow("toolBrush", displayImage);
    cvSetMouseCallback("toolBrush", toolBrush, NULL);
    while(true)
    {
        finImg.copyTo(displayImage);
        circle(displayImage, pointC, rad, Scalar(255, 255, 255), -1);
        imshow("toolBrush", displayImage);
        imshow("final", finImg);
        k = waitKey(30);
        dispval = disp.at<uchar>(point1);
        if (pointCdrawfg)
        {
            circle(fgRaw, pointC, rad, Scalar(255, 255, 255), -1);
            circle(displayDisp, pointC, rad+10, dispval, -1);
            updateFgBg(fgRaw, bgRaw, true, finImg, img1, disp);
        }
        if (pointCdrawbg)
        {
            circle(bgRaw, pointC, rad, Scalar(255, 255, 255), -1);
            updateFgBg(fgRaw, bgRaw, false, finImg, img1, disp);
        }
        imshow("newDisplayDisp", displayDisp);
        if (k ==27)
        {
            break;
        }
    }
    dispval = getThreshold(displayDisp, point1, 12, foreground);
    segmentForeground(img1, foreground, background, contours);
    Mat blurBackground;
    doMultiBlur(img1, blurBackground, disp, point1);
        //doMultiBlurHist(img1, blurBackground, disp, dispval);
    bitwise_and(background, blurBackground, background);
    getMaskedImage(img1, foreground);

    addFgBg(foreground, background, finImg);
    imshow("final Image refocus", finImg);
    waitKey(0);

    imwrite("newUpdatedfinaImage.jpg", finImg);
    imwrite("newUpdateddispImage.jpg", displayDisp);

    return(0);
    */
}

int updateFgBg(Mat fgRaw, Mat bgRaw, bool updateFg, Mat& finImg, Mat img1, Mat disp)
{
    //Mat bgRaw = Mat::ones(fgRaw.size(), CV_8UC3);
    if (updateFg)
    {
        bgRaw = Scalar(255, 255, 255) - fgRaw;
    }
    else
    {
        fgRaw = Scalar(255, 255, 255) - bgRaw;
    }
    Mat background, foreground;
    bgRaw.copyTo(background);
    fgRaw.copyTo(foreground);
    getSepia(img1, background);
    getMaskedImage(img1, foreground);
    //imshow("newfg", fgRaw);
    //imshow("newbg", bgRaw);
    addFgBg(foreground, background, finImg);
    return 1;
}

int getDisparity(Mat img, Mat& disp)
{
    Mat g1, g2;
    //cvtColor(img, img, CV_RGBA2BGR);
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    getDisp(g1, g2, disp);
    imwrite("refocusDisp.jpg", disp);
    
}

int fillDisp(Mat& disp)
{
    Mat inpaintMask;
    threshold(disp, inpaintMask, 10, 255, THRESH_BINARY_INV);
    inpaint(disp, inpaintMask, disp, 10, INPAINT_NS);
}

int getDisp(Mat g1, Mat g2, Mat &disp)
{
    Mat disp16;
    StereoSGBM sbm;
    sbm.SADWindowSize = 7; // 5
    sbm.numberOfDisparities = 112;
    sbm.preFilterCap = 20;
    sbm.minDisparity = -64; // -64
    sbm.uniquenessRatio = 1; // 1
    sbm.speckleWindowSize = 120; //150
    sbm.speckleRange = 2;
    sbm.disp12MaxDiff = 10; // 10
    sbm.fullDP = false;
    sbm.P1 = 600;
    sbm.P2 = 2400;
    sbm(g1, g2, disp16);
    normalize(disp16, disp, 0, 255, CV_MINMAX, CV_8U);
    medianBlur(disp, disp, 5);
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
    range = disval/10;
    inRange(img, disval - range, disval + range, foreground);
    medianBlur(foreground, foreground, 9);
    return 1;
}

int getThresholdHist(Mat img, int dispval, int range, Mat &foreground)
{
    range = dispval/10;
    inRange(img, dispval - range, dispval + range, foreground);
    medianBlur(foreground, foreground, 9);
    return 1;
}

int segmentForeground(Mat img, Mat &foreground, Mat &background, vector<vector<Point> >& contours)
{
    //vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat drawing, kernel;
    int size=3;
    drawing = Mat::zeros(img.size(), CV_8UC3);
    findContours(foreground.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 15000)
        {
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
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    foreground = drawing.clone();
    background = Scalar(255, 255, 255) - foreground;
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
    return 1;
}
int getBlurMaskedImage(Mat img, Mat &foreground)
{
    Mat blur;
    GaussianBlur(img, blur, Size(13, 13), 15, 15);
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

int deFocus(Mat img, Mat& foreground, int size, int radius)
{
    Mat filter, blur;
    float totalSum;
    filter = Mat::zeros(size, size, CV_64F);
    circle(filter, Point(size/2, size/2), radius, Scalar(1, 1, 1), -1);
    totalSum = sum(filter)[0];
    filter = filter/totalSum;
    filter2D(img, blur, -1, filter);
    bitwise_and(blur, foreground, foreground);
    return 1;
}

int doBokehImg(Mat disp, Mat img, Mat& foreground)
{
    int i, j, disval, size=30, dia;
    float tSum;
    Mat cImg;
    Mat bImg, filter;
    cImg = img.clone();
    bImg = img.clone();

    for(i=0; i<img.rows-size; i+=size/2)
    {
        for(j=0; j<img.cols-size; j+=size/2)
        {
            Mat subDisp = disp(Range(i, i+size), Range(j, j+size));
            Mat subImg = cImg(Range(i, i+size), Range(j, j+size));
            disval = sum(subDisp)[0]/(size*size);
            dia = 13 - disval/20;
            if (dia < 2)
                continue;
            filter = Mat::zeros(15, 15, CV_64F);
            //printf("(%d, %d) %d %d\n", i, j, dia, disval);
            circle(filter, Point(7, 7), dia/2+1, (1, 1, 1), -1);
            tSum = sum(filter)[0];
            filter = filter/tSum;
            filter2D(subImg, subImg, -1, filter);
            //subImg.copyTo(bImg(Rect(i, j, size, size)));
            subImg.copyTo(bImg.colRange(j, j+size).rowRange(i, i+size));
            //bImg(Range(i, i+size), Range(j, j+size)) = subImg;
        }
    }
    //imshow("bokeh", bImg);
    bitwise_and(bImg, foreground, foreground);
    return 1;
}

int getSepia(Mat img, Mat &foreground)
{
    Mat cImg, kernel;
    kernel = (cv::Mat_<float>(3,3) <<  0.272, 0.534, 0.131,
                                        0.349, 0.686, 0.168,
                                        0.393, 0.769, 0.189);//,
                                        //0, 0, 0, 1);
    transform(img, cImg, kernel);
    bitwise_and(cImg, foreground, foreground);
    return 1;
}

int doBokehImgRelative(Mat disp, Mat img, Mat& foreground, Point p1)
{
    int dispVal;
    dispVal = disp.at<uchar>(p1.y, p1.x);
    int i, j, disval, size=30, dia, diff;
    float tSum;
    Mat cImg;
    Mat bImg, filter;
    cImg = img.clone();
    bImg = img.clone();

    for(i=0; i<img.rows-size; i+=size/2)
    {
        for(j=0; j<img.cols-size; j+=size/2)
        {
            Mat subDisp = disp(Range(i, i+size), Range(j, j+size));
            Mat subImg = cImg(Range(i, i+size), Range(j, j+size));
            disval = sum(subDisp)[0]/(size*size);
            diff = abs(dispVal - disval);
            if (diff < 12)
                continue;
            dia = diff/10;
            if (dia < 2)
                continue;
            filter = Mat::zeros(15, 15, CV_64F);
            //printf("(%d, %d) %d %d\n", i, j, dia, disval);
            circle(filter, Point(7, 7), dia/2+1, (1, 1, 1), -1);
            tSum = sum(filter)[0];
            filter = filter/tSum;
            filter2D(subImg, subImg, -1, filter);
            //subImg.copyTo(bImg(Rect(i, j, size, size)));
            subImg.copyTo(bImg.colRange(j, j+size).rowRange(i, i+size));
            //bImg(Range(i, i+size), Range(j, j+size)) = subImg;
        }
    }
    //bImg.copyTo(foreground);
    //imshow("bokeh", bImg);
    imwrite("bokehImageRelative.jpg", bImg);
    bitwise_and(bImg, foreground, foreground);
    return 1;
}

int doMultiBlur(Mat img, Mat& retVal, Mat disp, Point p1)
{
    int dispval, range, i, lval, hval;
    int l1, l2, h1, h2;
    vector<Mat> layers, blurs, finLayers;
    //Mat thresh, blur, bitwiseImg;
    int threshVal;

    dispval = disp.at<uchar>(p1.y, p1.x);
    range = dispval/10;
    //printf("%d %d\n", range, dispval);
    lval = dispval+1;
    hval = dispval-1;
    for(i=1; i<4; i++)
    {
        l1 = lval - range;
        l2 = lval;
        h1 = hval;
        h2 = hval + range;
        //printf("%d %d %d %d\n", l1, l2, h1, h2);
        Mat thresh;
        Mat seg;
        threshVal = getThresh(disp, thresh, l1, l2, h1, h2);
        if (!threshVal)
        {
            //printf("break\n");
            break;
        }
        thresh.copyTo(seg);
        segmentBlurs(thresh, seg);
        imshow("thresh", thresh);
        imshow("seg", seg);
        waitKey(0);
        layers.push_back(seg);

        lval = l1;
        hval = h2;
        range*=2;
    }

    blurs.push_back(img);
    for(i=1; i<layers.size(); i++)
    {
        Mat blur;
        GaussianBlur(img, blur, Size(19, 19), 2*i);
        //doCircBlur(img, blur, 3*i);
        //imshow("blur", blur);
        //waitKey(0);
        blurs.push_back(blur);
    }
    int sigma = 2*i;
    Mat backLayer;
    backLayer = Mat::zeros(img.rows, img.cols, CV_8UC3);
    for(i=1; i<layers.size(); i++)
    {
        Mat bitwiseImg;
        //printf("%d %d %d %d\n", layers[i].cols, layers[i].rows, blurs[i].rows, blurs[i].cols);
        //printf("%d %d\n", layers[i].channels(), blurs[i].channels());
        bitwise_and(layers[i], blurs[i], bitwiseImg);
        imshow("bitwiseImg", bitwiseImg);
        waitKey(0);
        //printf("%d %d %d %d\n", layers[i].cols, layers[i].rows, backLayer.rows, backLayer.cols);
        add(backLayer, layers[i], backLayer);
        //imshow("thresh", layers[i]);
        
        finLayers.push_back(bitwiseImg);
    }
    //imshow("backLayer", backLayer);
    //waitKey(0);
    Mat blurImage;
    backLayer = Scalar(255, 255, 255) - backLayer;
    GaussianBlur(img, blurImage, Size(19, 19), sigma);
    bitwise_and(blurImage, backLayer, backLayer);
    finLayers.push_back(backLayer);
    stackUp(finLayers, retVal);

    return 1;

}

int getThresh(Mat img, Mat& retVal, int l1, int l2, int h1, int h2)
{
    Mat thresh1, thresh2, thresh;
    if (l2 < 0 && h1 > 255)
    {
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
    Mat zerotemp;
    zerotemp = Mat::zeros(layers[i].size(), CV_8UC3);
    zerotemp.copyTo(retVal);
    //retVal = Mat::zeros(layers[i].size(), CV_8UC3);
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
    if (disp.cols > 0 && disp.rows > 0)
    {
        return 1;
    }
    return 0;
}

int segmentBlurs(Mat img, Mat &foreground)
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
        if (contourArea(contours[i]) > 2000)
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
        if (contourArea(contours[i]) > 5000)
        {
            printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    //dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    foreground = drawing.clone();
    //imshow("contours", drawing);
    //background = Scalar(255, 255, 255) - foreground;

    drawing = Mat::zeros(img.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    return 1;
}

int doOilPaint(Mat src, Mat& foreground)
{
    Mat dst;
    int nRadius = 5;
    int fIntensityLevels = 20;

    dst = Mat::zeros(src.size(), src.type());

    for(int nY = nRadius; nY < (src.rows - nRadius); nY++)
    {
        for(int nX = nRadius; nX < (src.cols - nRadius); nX++)
        {
            int nIntensityCount[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            int nSumB[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            int nSumG[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            int nSumR[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            for(int nY_O = -nRadius; nY_O <= nRadius; nY_O++)
            {
                for(int nX_O = -nRadius; nX_O <= nRadius; nX_O++)
                {
                    int nB = src.at<Vec3b>((nY+nY_O), (nX+nX_O))[0];
                    int nG = src.at<Vec3b>((nY+nY_O), (nX+nX_O))[1];
                    int nR = src.at<Vec3b>((nY+nY_O), (nX+nX_O))[2];

                    int nCurIntensity = (((nB+nG+nR)/3.0)*fIntensityLevels)/255;
                    if(nCurIntensity > 255)
                    {
                        nCurIntensity = 255;
                    }
                    int i = nCurIntensity;
                    nIntensityCount[i]++;

                    nSumB[i] = nSumB[i] + nB;
                    nSumG[i] = nSumG[i] + nG;
                    nSumR[i] = nSumR[i] + nR;

                }
            }

            int nCurMax = 0;
            int nMaxIndex = 0;

            for(int nI = 0; nI<21; nI++)
            {
                if(nIntensityCount[nI] > nCurMax)
                {
                    nCurMax = nIntensityCount[nI];
                    nMaxIndex = nI;
                }
            }

            dst.at<Vec3b>(nY, nX)[0] = nSumB[nMaxIndex]/nCurMax;
            dst.at<Vec3b>(nY, nX)[1] = nSumG[nMaxIndex]/nCurMax;
            dst.at<Vec3b>(nY, nX)[2] = nSumR[nMaxIndex]/nCurMax;
        }
    }
    bitwise_and(dst, foreground, foreground);
    return 1;
}

int doGraySingle(Mat img, Mat& retVal, Mat disp, Point p1)
{
    int dispval, range;
    int l1, l2, h1, h2;
    dispval = disp.at<uchar>(p1.y, p1.x);
    range = getRange(disp, p1);

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

    bitwise_and(img, thresh, retVal);

    thresh = Scalar(255, 255, 255) - thresh;

    Mat gray;
    cvtColor(img, gray, CV_BGR2GRAY);

    vector<Mat> grayvec;
    grayvec.push_back(gray);
    grayvec.push_back(gray);
    grayvec.push_back(gray);
    merge(grayvec, gray);
    bitwise_and(gray, thresh, gray);
    add(gray, retVal, retVal);
    return 1;
}

int getRange(Mat disp, Point p1)
{
    int dispval, range;
    dispval = disp.at<uchar>(p1.y, p1.x);
    range = dispval/8;
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
            //printf("%lf\n", contourArea(contours[i]));
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
            //printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    //dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    //foreground = drawing.clone();
    //imshow("contours", drawing);
    //background = Scalar(255, 255, 255) - foreground;

    Mat dispClone;
    bitwise_and(disp, drawing, dispClone);
    //imshow("dispClone", dispClone);
    Point minP, maxP;
    double minVal, maxVal;

    minMaxLoc(dispClone, &minVal, &maxVal, &minP, &maxP, thresh);
    //printf("%lf %lf\n", minVal, maxVal);

    //drawing = Mat::zeros(thresh.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    return (maxVal - dispval);
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

int histPick(Mat disp)
{
    
    int pxv[256] = {0};

    for(int i=0; i<disp.rows; i++)
    {
        for(int j=0; j<disp.cols; j++)
        {
            pxv[disp.at<uchar>(j, i)] += 1;
        }
    }

    int maxval=0;
    int val;
    int maxindex = 255;

    for(int i=230; i>100; i--)
    {
        val = pxv[i];
        if (val > maxval)
        {
            maxval = val;
            maxindex = i;
        }
    }
    return (maxindex);
}

int doMultiBlurHist(Mat img, Mat& retVal, Mat disp, int dispval)
{
    int range, i, lval, hval;
    int l1, l2, h1, h2;
    vector<Mat> layers, blurs, finLayers;
    //Mat thresh, blur, bitwiseImg;
    int threshVal;

    range = dispval/10;
    //printf("%d %d\n", range, dispval);
    lval = dispval+1;
    hval = dispval-1;
    for(i=1; i<4; i++)
    {
        l1 = lval - range;
        l2 = lval;
        h1 = hval;
        h2 = hval + range;
        //printf("%d %d %d %d\n", l1, l2, h1, h2);
        Mat thresh;
        Mat seg;
        threshVal = getThresh(disp, thresh, l1, l2, h1, h2);
        if (!threshVal)
        {
            //printf("break\n");
            break;
        }
        thresh.copyTo(seg);
        segmentBlurs(thresh, seg);
        imshow("thresh", thresh);
        imshow("seg", seg);
        waitKey(0);
        layers.push_back(seg);

        lval = l1;
        hval = h2;
        range*=2;
    }

    blurs.push_back(img);
    for(i=1; i<layers.size(); i++)
    {
        Mat blur;
        GaussianBlur(img, blur, Size(19, 19), 2*i);
        //doCircBlur(img, blur, 3*i);
        //imshow("blur", blur);
        //waitKey(0);
        blurs.push_back(blur);
    }
    int sigma = 2*i;
    Mat backLayer;
    backLayer = Mat::zeros(img.rows, img.cols, CV_8UC3);
    for(i=1; i<layers.size(); i++)
    {
        Mat bitwiseImg;
        //printf("%d %d %d %d\n", layers[i].cols, layers[i].rows, blurs[i].rows, blurs[i].cols);
        //printf("%d %d\n", layers[i].channels(), blurs[i].channels());
        bitwise_and(layers[i], blurs[i], bitwiseImg);
        imshow("bitwiseImg", bitwiseImg);
        waitKey(0);
        //printf("%d %d %d %d\n", layers[i].cols, layers[i].rows, backLayer.rows, backLayer.cols);
        add(backLayer, layers[i], backLayer);
        //imshow("thresh", layers[i]);
        
        finLayers.push_back(bitwiseImg);
    }
    //imshow("backLayer", backLayer);
    //waitKey(0);
    Mat blurImage;
    backLayer = Scalar(255, 255, 255) - backLayer;
    GaussianBlur(img, blurImage, Size(19, 19), sigma);
    bitwise_and(blurImage, backLayer, backLayer);
    finLayers.push_back(backLayer);
    stackUp(finLayers, retVal);

    return 1;

}