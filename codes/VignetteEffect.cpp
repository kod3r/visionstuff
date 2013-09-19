#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <math.h>
using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }
    
    Mat src;
    Mat final;
    
    src = imread(argv[1], 1);
    
    if(! src.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    Mat gaussimage;
    Mat gray;
    gaussimage = Mat::zeros(src.rows, src.cols, CV_32F);
    float d, val;
    float dia = sqrt(src.cols * src.cols + src.rows * src.rows);
    int order=3;
    for(int i=0; i<src.rows; i++)
    {
        for(int j=0; j<src.cols; j++)
        {
            d = sqrt((j-src.cols/2)*(j-src.cols/2)+(i-src.rows/2)*(i-src.rows/2));
            //val = (255*exp(-(d*d)/((dia*dia)*2)));
            val = 1;
            for(int k=0; k<order; k++)
                val = val /(1+(d/dia));
            val = 255*val;
            gaussimage.at<float>(i,j)=val;
        }
    }
    imshow("gaussF", gaussimage);
    gaussimage.convertTo(gaussimage, CV_8U);
    //gaussimage = gaussimage/2;
    cvtColor(src, gray, CV_BGR2GRAY);
    gaussimage = Mat::zeros(src.rows, src.cols, CV_8UC3);
    dia = sqrt(src.cols*src.cols+src.rows*src.rows);
    circle(gaussimage, Point(src.cols/2, src.rows/2), dia/2-30, Scalar(255, 255, 255), -1);
    GaussianBlur(gaussimage, gaussimage, Size(81,81), 5);
    //vector<Mat> gauss;
    //gauss.push_back(gaussimage);
    //gauss.push_back(gaussimage);
    //gauss.push_back(gaussimage);
    //merge(gauss, gaussimage);
    //gray.convertTo(gray, CV_32F);
    imshow("before", src);
    addWeighted(src, 0.9, gaussimage, 0.08, 1.5, src);
    //bitwise_and(gaussimage, src, src);

    imshow("gauss", gaussimage);
    imshow("after", src);
    imwrite("vignetteEffect.jpg", src);
    waitKey(0);
}
