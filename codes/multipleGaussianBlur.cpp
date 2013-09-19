#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
using namespace std;
using namespace cv;

int getMultiplePartialBlur(Mat img, Mat& retVal, Point p1, int radius, int n);
double getMaxDist(Point p1, int w, int h);
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
    getMultiplePartialBlur(src, retVal, Point(192, 85), 100, 4);
    //GaussianBlur(retVal, retVal, Size(3,3), 0);
    imshow("image", src);
    imshow("blur", retVal);
    waitKey(0);
    imwrite("multipleGaussBlur.jpg", retVal);
    return 1;
}

int getMultiplePartialBlur(Mat img, Mat& retVal, Point p1, int radius, int n)
{
    Mat blur;
    vector<Mat> circmask, layers;

    Mat circ;
    Mat layer;
    Mat invcircmask;
    circ = Mat::zeros(img.rows, img.cols, CV_8UC3);
    retVal = Mat::zeros(img.rows, img.cols, CV_8UC3);
    circle(circ, p1, radius, Scalar(255, 255, 255), -1);
    //GaussianBlur(img, blur, Size(1,1), 1);
    bitwise_and(img, circ, layer);
    circmask.push_back(circ.clone());
    layers.push_back(layer.clone());

    
    double maxDist = getMaxDist(p1, img.cols, img.rows);
    printf("max = %lf\n", maxDist);
    int new_radius = int(maxDist) - radius;
    printf("new_radius = %d\n", new_radius);
    new_radius = double(new_radius)/double(pow(2, n-1));
    new_radius = int(new_radius);
    printf("new_radius = %d\n", new_radius);
    int blurSize = 3;
    for (int i=0; i<n; i++)
    {
        GaussianBlur(img, blur, Size(blurSize, blurSize), 0);
        circle(circ, p1, radius + new_radius, Scalar(255, 255, 255), -1);
        for(int j=0; j<circmask.size(); j++)
        {
            circ = circ - circmask[j];
        }
        bitwise_and(blur, circ, layer);
        layers.push_back(layer.clone());
        circmask.push_back(circ.clone());
        new_radius *= 2;
        blurSize += 2;
    }
    layer = Mat::zeros(img.rows, img.cols, CV_8UC3);
    //circ = Mat::zeros(img.rows, img.cols, CV_8UC3);
    for(int k=0; k<layers.size(); k++)
    {
        add(retVal, layers[k], retVal);
        imshow("layer", layers[k]);
        waitKey(0);
        //circ += circmask[k];
    }
    //circ = Scalar(255, 255, 255) - circ;
    //GaussianBlur(img, blur, Size(blurSize, blurSize), 0);
    //bitwise_and(blur, circ, layer);
    //add(retVal, layer, retVal);
    return 1;
}

double getMaxDist(Point p1, int w, int h)
{
    double maxVal;
    if (p1.x < w/2 && p1.y < h/2)
        maxVal = sqrt((p1.x-w)*(p1.x-w) + (p1.y-h)*(p1.y-h));
    else if (p1.x > w/2 && p1.y < h/2)
        maxVal = sqrt((p1.x)*(p1.x) + (p1.y-h)*(p1.y-h));
    else if (p1.x > w/2 && p1.y > h/2)
        maxVal = sqrt((p1.x)*(p1.x) + (p1.y)*(p1.y));
    else
        maxVal = sqrt((p1.x-w)*(p1.x-w) + (p1.y)*(p1.y));
    return maxVal;
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