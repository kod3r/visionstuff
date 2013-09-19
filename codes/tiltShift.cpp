#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <iostream>
#include <math.h>
using namespace std;
using namespace cv;

int getTiltBlur(Mat img, Mat& retVal, Point p1, int height, float angle);
int addFgBg(Mat foreground, Mat background, Mat &img);
int getave(Mat &img, int x, int y, int size);

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }
    Mat src;
    src = imread(argv[1]);
    if(! src.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    Mat retVal;
    getTiltBlur(src, retVal, Point(237, 293), 100, 135.0);
    //GaussianBlur(retVal, retVal, Size(3,3), 0);
    imshow("image", src);
    imshow("blur", retVal);
    waitKey(0);
    imwrite("tiltBlur.jpg", retVal);
    return 1;
}

int getTiltBlur(Mat img, Mat& retVal, Point p1, int height, float angle)
{
    Mat blur;
    Mat layer, blurLayer;
    Mat rect, invrect;
    Mat rotMat;
    Point tlp, brp;

    tlp = Point(p1.y - height/2, 0);
    brp = Point(p1.y + height/2, img.cols);

    rect = Mat::zeros(img.rows, img.cols, CV_8UC3);
    rectangle(rect, tlp, brp, Scalar(255, 255, 255), -1);
    rotMat = getRotationMatrix2D(p1, angle, 1.0);
    warpAffine(rect, rect, rotMat, img.size());
    invrect = Scalar(255, 255, 255) - rect;
    bitwise_and(img, rect, layer);
    GaussianBlur(img, blur, Size(3,3), 0);
    bitwise_and(blur, invrect, blurLayer);
    addFgBg(layer, blurLayer, retVal);
    imshow("tilted", rect);
    waitKey(0);
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