#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/photo/photo.hpp"
#include <stdio.h>
#include <string.h>
#include <math.h>

using namespace cv;
using namespace std;
Point point1;

int getDisparity(Mat g1, Mat g2, Mat &disp);
int getThreshold(Mat img, Point p1, int range, Mat &foreground);
int segmentForeground(Mat img, Mat &foreground, Mat &background);
int getBlurMaskedImage(Mat img, Mat &foreground);
int getMaskedImage(Mat img, Mat &foreground);
int addFgBg(Mat foreground, Mat background, Mat &img);
int getBlurMaskedGrayImage(Mat img, Mat &foreground);
int getMaskedGrayImage(Mat img, Mat &foreground);
int filterDisp(Mat &img);
int deFocus(Mat img, Mat& foreground, int size, int radius);
int deFocus8(Mat img, Mat& foreground, int size, int w, int h);
int doBokeh(Mat disp, Mat img);
int doBokehImg(Mat disp, Mat img, Mat& foreground);
int getSepia(Mat img, Mat &foreground);
int getPartialBlur(Mat img, Point p1, int radius, Mat &foreground);
int getPartialErodeBlur(Mat img, Mat &foreground);
int getMultiplePartialBlur(Mat img, Point p1, int radius, Mat& foreground);
int radialBlur(Mat img, Point p1, Mat &foreground);
int radialBlurGrad(Mat img, Point p1, Mat& foreground);
int vignetteEffect(Mat img, Point p1, Mat &foreground);
int doGrabCut(Mat img, Mat& retVal, Point p1, Point p2);
int doBokehImgRelative(Mat disp, Mat img, Mat& foreground, Point p1);
int doOilPaint(Mat src, Mat& foreground);
int addFgBgAlpha(Mat foreground, Mat background, Mat &img);

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
    //cvtColor(img, img, CV_RGBA2BGR);
    //resize(img, img, Size(img.cols/3, img.rows/3));
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));

    //resize(img1, img1, Size(620, 360));
    //resize(img2, img2, Size(620, 360));
    //resize(img1, img1, Size(560, 630));
    //resize(img2, img2, Size(560, 630));
    printf("%d %d\n", img2.cols, img2.rows);
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    imshow("img", img1);
    cvSetMouseCallback("img", mouseHandler, NULL);
    waitKey(0);

    getDisparity(g1, g2, disp);
    //doBokeh(disp, img1);
    imwrite("disp.jpg", disp);
    imshow("disparity", disp);
    //filterDisp(disp); // never do this
    getThreshold(disp, point1, 16, foreground);
    vector<Mat> fv;
    fv.push_back(foreground);
    fv.push_back(foreground);
    fv.push_back(foreground);
    merge(fv, foreground);
    imwrite("foregrounddisp.jpg", foreground);
    background = Scalar(255, 255, 255) - foreground;
    //segmentForeground(img1, foreground, background);
    if (fg == 1)
    {
        //getBlurMaskedImage(img1, background);
        //deFocus8(img1, background, 20, 7, 7);
        //deFocus(img1, background, 20, 10);
        //doBokehImg(disp, img1, background);
        doBokehImgRelative(disp, img1, background, point1);
        //doOilPaint(img1, background);
        //getSepia(img1, background);
        //getPartialBlur(img1, point1, 150, background);
        //getMultiplePartialBlur(img1, point1, 100, background);
        //radialBlurGrad(img1, point1, background);
        //vignetteEffect(img1, point1, background);
        //getPartialErodeBlur(img1, background);
        //getBlurMaskedImage(img1, background);
        //getBlurMaskedGrayImage(img1, background);
        //getMaskedGrayImage(img1, background);
        getMaskedImage(img1, foreground);
        //addFgBgAlpha(foreground, background, finImg);
        addFgBg(foreground, background, finImg);
        //resize(finImg, finImg, Size(1280, 720));
        //imwrite("backgroundBokeh.jpg", finImg);
        //imwrite("backgroundDeFocus8.jpg", finImg);
        //imwrite("backgroundDeFocus.jpg", finImg);
        //imwrite("backgroundOilPaint.jpg", finImg);
        //imwrite("BackgroundpartialDilateBlur.jpg", finImg);
        //imwrite("BackgroundMultiplePartialBlur.jpg", finImg);
        //imwrite("BackgroundradialBlurGrad.jpg", finImg);
    }
    else
    {
        //getBlurMaskedImage(img1, foreground);
        //getBlurMaskedGrayImage(img1, foreground);
        getMaskedGrayImage(img1, foreground);
        getMaskedImage(img1, background);
        addFgBg(foreground, background, finImg);
        imwrite("foregorundgray.jpg", finImg);
    }
    imshow("foreground", foreground);
    imshow("background", background);
    imshow("finImg", finImg);
    cvtColor(finImg, finImg, CV_BGR2GRAY);
    imwrite("grayimage.jpg", finImg);
    waitKey(0);
    return 1;
}

int getDisparity(Mat g1, Mat g2, Mat &disp)
{
    Mat disp16;
    StereoSGBM sbm;
    sbm.SADWindowSize = 7; // 5
    sbm.numberOfDisparities = 128; //192 //128
    sbm.preFilterCap = 15;
    sbm.minDisparity = -39; // -64 // -39
    sbm.uniquenessRatio = 1; // 1
    sbm.speckleWindowSize = 150; //150
    sbm.speckleRange = 5;
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
    //Mat newDisp(disp, Rect(70, 100, 500, 500));
    Mat newDisp;
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
    double min, max;
    Point minP, maxP;
    minMaxLoc(img, &min, &max, &minP, &maxP);
    printf("max=%lf\n", max);
    printf("disparityVal = %d\n", disval);
    range = disval/15;
    printf("range=%d\n", range);
    inRange(img, disval - range, disval + range, foreground);
    imshow("thresh", foreground);
    //medianBlur(foreground, foreground, 9);
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
    /*
    Mat grabcutimg;
    //doGrabCut(temp, grabcutimg, Point(360, 75), Point(450, 620));
    doGrabCut(temp, grabcutimg, Point(187, 55), Point(340, 606));
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
    */
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
    GaussianBlur(img, blur, Size(13, 13), 13, 13);
    blur.copyTo(foreground);
    //bitwise_and(blur, foreground, foreground);
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

int addFgBgAlpha(Mat foreground, Mat background, Mat &img)
{
    Mat tempImg;
    addWeighted(foreground, 0.5, background, 0.8, 0.5, img);
    return 1;
    Mat r, g, b;
    vector<Mat> gray;
    gray.push_back(b);
    gray.push_back(g);
    gray.push_back(r);
    split(img, gray);
    equalizeHist(b, b);
    equalizeHist(g, g);
    equalizeHist(r, r);
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

int deFocus8(Mat img, Mat& foreground, int size, int w, int h)
{
    Mat filter1, filter2, rotMat, blur;
    float totalSum;
    filter1 = Mat::zeros(size, size, CV_64F);
    rectangle(filter1, Point(size/2-w/2, size/2-h/2),
              Point(size/2+w/2, size/2+h/2),
              Scalar(255, 255, 255), -1);
    rotMat = getRotationMatrix2D(Point(size/2, size/2), 45.0, 1.0);
    warpAffine(filter1, filter2, rotMat, Size(size, size));
    bitwise_or(filter1, filter2, filter1);
    filter1 = filter1/255.0;
    totalSum = sum(filter1)[0];
    filter1 = filter1/totalSum;
    filter2D(img, blur, -1, filter1);
    bitwise_and(blur, foreground, foreground);
    return 1;

}

int doBokeh(Mat disp, Mat img)
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
            printf("(%d, %d) %d %d\n", i, j, dia, disval);
            circle(filter, Point(7, 7), dia/2+1, (1, 1, 1), -1);
            tSum = sum(filter)[0];
            filter = filter/tSum;
            filter2D(subImg, subImg, -1, filter);
            //subImg.copyTo(bImg(Rect(i, j, size, size)));
            subImg.copyTo(bImg.colRange(j, j+size).rowRange(i, i+size));
            //bImg(Range(i, i+size), Range(j, j+size)) = subImg;
        }
    }
    imshow("bokeh", bImg);
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
    imwrite("bokehImage.jpg", bImg);
    bitwise_and(bImg, foreground, foreground);
    return 1;    
}

int doBokehImgRelative(Mat disp, Mat img, Mat& foreground, Point p1)
{
    int dispVal;
    dispVal = disp.at<uchar>(p1.y, p1.x);
    int i, j, disval, size=10, dia, diff;
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
            //printf("%d %d (%d, %d)\n", dia, diff, j, i);
            if (diff < 5)
                continue;
            dia = diff/10;
            //if (dia < 2)
                //continue;
            
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
    imshow("bokeh", bImg);
    imwrite("bokehImageRelative.jpg", bImg);
    bitwise_and(bImg, foreground, foreground);
    //bImg.copyTo(foreground);
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

int getPartialBlur(Mat img, Point p1, int radius, Mat &foreground)
{
    Mat blur, circmask, invcircmask, mask;
    GaussianBlur(img, blur, Size(15, 15), 0, 0);
    circmask = Mat::zeros(img.rows, img.cols, CV_8UC3);
    circle(circmask, p1, radius, Scalar(255, 255, 255), -1);
    invcircmask = Scalar(255, 255, 255) - circmask;
    bitwise_and(invcircmask, blur, invcircmask);
    GaussianBlur(img, blur, Size(5, 5), 0, 0);
    bitwise_and(circmask, blur, circmask);
    mask = circmask + invcircmask;
    bitwise_and(mask, foreground, foreground);
    return 1;
}

int getMultiplePartialBlur(Mat img, Point p1, int radius, Mat& foreground)
{
    Mat blur, circmask, layer1, mask;
    Mat circmask1, layer2;
    Mat circmask2, layer3;
    Mat invcircmask, layer4;
    GaussianBlur(img, blur, Size(5, 5), 0, 0);
    circmask = Mat::zeros(img.rows, img.cols, CV_8UC3);
    circle(circmask, p1, radius/3, Scalar(255, 255, 255), -1);
    bitwise_and(blur, circmask, layer1);
    GaussianBlur(img, blur, Size(7, 7), 0, 0);
    circmask1 = Mat::zeros(img.rows, img.cols, CV_8UC3);
    circle(circmask1, p1, 2*radius/3, Scalar(255, 255, 255), -1);
    circmask1 = circmask1 - circmask;
    bitwise_and(blur, circmask1, layer2);
    GaussianBlur(img, blur, Size(9, 9), 0, 0);
    circmask2 = Mat::zeros(img.rows, img.cols, CV_8UC3);
    circle(circmask2, p1, radius, Scalar(255, 255, 255), -1);
    invcircmask = Scalar(255, 255, 255) - circmask2;
    circmask2 = circmask2 - circmask1;
    circmask2 = circmask2 - circmask;
    bitwise_and(blur, circmask2, layer3);
    GaussianBlur(img, blur, Size(13, 13), 0, 0);
    bitwise_and(blur, invcircmask, layer4);
    mask = layer1 + layer2 + layer3 + layer4;
    bitwise_and(mask, foreground, foreground);
    //mask.copyTo(foreground);
    return 1;
    
}

int getPartialErodeBlur(Mat img, Mat &foreground)
{
    Mat kernel, mask, blur, blurmask, dilmask, invdilmask;
    int size=3;
    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));
    erode(foreground, dilmask, kernel, Point(-1, -1), 2);
    invdilmask = Scalar(255, 255, 255) - dilmask;
    GaussianBlur(img, blur, Size(15, 15), 0, 0);
    bitwise_and(dilmask, blur, dilmask);
    GaussianBlur(img, blur, Size(5, 5), 0, 0);
    bitwise_and(invdilmask, blur, invdilmask);
    mask = dilmask + invdilmask;
    imshow("dilmask", dilmask);
    imshow("invdilmask", invdilmask);
    bitwise_and(mask, foreground, foreground);
    return 1;
}

int radialBlur(Mat img, Point p1, Mat &foreground)
{
    int width, height;
    width = img.cols;
    height = img.rows;
    float center_x = p1.x; //or whatever
    float center_y = p1.y;
    float blur = 0.007; //blur radius per pixels from center. 2px blur at 1000px from center
    int iterations = 5;

    Mat growMapx, growMapy;
    Mat shrinkMapx, shrinkMapy;
    growMapx = Mat::zeros(height, width, CV_32F);
    growMapy = Mat::zeros(height, width, CV_32F);
    shrinkMapx = Mat::zeros(height, width, CV_32F);
    shrinkMapy = Mat::zeros(height, width, CV_32F);
    for(int x = 0; x < width; x++) {
      for(int y = 0; y < height; y++) {
        growMapx.at<float>(y,x) = x+((x - center_x)*blur);
        growMapy.at<float>(y,x) = y+((y - center_y)*blur);
        shrinkMapx.at<float>(y,x) = x-((x - center_x)*blur);
        shrinkMapy.at<float>(y,x) = y-((y - center_y)*blur);
      }
    }
    Mat mask;
    Mat tmp1, tmp2;
    for(int i = 0; i < iterations; i++)  {
      remap(img, tmp1, growMapx, growMapy, CV_INTER_LINEAR); // enlarge
      remap(img, tmp2, shrinkMapx, shrinkMapy, CV_INTER_LINEAR); // shrink
      addWeighted(tmp1, 0.5, tmp2, 0.5, 0, mask); // blend back to src
    }
    //GaussianBlur(mask, mask, Size(5,5), 0, 0);
    bitwise_and(mask, foreground, foreground);
    return 1;
}

int radialBlurGrad(Mat img, Point p1, Mat& foreground)
{
    int width, height;
    width = img.cols;
    height = img.rows;
    float center_x = p1.x; //or whatever
    float center_y = p1.y;
    float blur = 0.000055; //blur radius per pixels from center. 2px blur at 1000px from center
    float blurRadius = 0;
    int iterations = 5;

    Mat growMapx, growMapy;
    Mat shrinkMapx, shrinkMapy;
    growMapx = Mat::zeros(height, width, CV_32F);
    growMapy = Mat::zeros(height, width, CV_32F);
    shrinkMapx = Mat::zeros(height, width, CV_32F);
    shrinkMapy = Mat::zeros(height, width, CV_32F);
    for(int x = 0; x < width; x++) {
      for(int y = 0; y < height; y++) {
        blurRadius = blur*sqrt((center_x-x)*(center_x-x)+(center_y-y)*(center_y-y));
        growMapx.at<float>(y,x) = x+((x - center_x)*blurRadius);
        growMapy.at<float>(y,x) = y+((y - center_y)*blurRadius);
        shrinkMapx.at<float>(y,x) = x-((x - center_x)*blurRadius);
        shrinkMapy.at<float>(y,x) = y-((y - center_y)*blurRadius);
      }
    }
    Mat mask;
    Mat tmp1, tmp2;
    for(int i = 0; i < iterations; i++)  {
      remap(img, tmp1, growMapx, growMapy, CV_INTER_LINEAR); // enlarge
      remap(img, tmp2, shrinkMapx, shrinkMapy, CV_INTER_LINEAR); // shrink
      addWeighted(tmp1, 0.5, tmp2, 0.5, 0, mask); // blend back to src
    }
    //GaussianBlur(mask, mask, Size(5,5), 0, 0);
    bitwise_and(mask, foreground, foreground);
    return 1;
}

int vignetteEffect(Mat img, Point p1, Mat &foreground)
{
    Mat gaussimage, mask;
    gaussimage = Mat::zeros(img.rows, img.cols, CV_8UC3);
    float dia;
    dia = sqrt(img.cols*img.cols+img.rows*img.rows);
    circle(gaussimage, Point(p1.x, p1.y), dia/2-30, Scalar(255, 255, 255), -1);
    GaussianBlur(gaussimage, gaussimage, Size(81,81), 5);
    addWeighted(img, 0.9, gaussimage, 0.08, 0, mask);
    printf("%d %d\n", mask.rows, mask.cols);
    printf("%d %d\n", foreground.rows, foreground.cols);
    bitwise_and(mask, foreground, foreground);
    return 1;
}

int doOilPaint(Mat src, Mat& foreground)
{
    Mat dst;
    int nRadius = 5;
    int fIntensityLevels = 20;
    
    //dst = src.clone();
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