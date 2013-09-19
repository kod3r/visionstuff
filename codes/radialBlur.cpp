#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdio.h>
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
    src = imread(argv[1]);
    if(! src.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    int width, height;
    width = src.cols;
    height = src.rows;
    printf("%d %d\n", width, height);
    float center_x = 275; //or whatever
    float center_y = 389;
    float blur = 0.00001; //blur radius per pixels from center. 2px blur at 1000px from center
    float blurRadius = 0;
    int iterations = 3;

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

    Mat tmp1, tmp2;
    for(int i = 0; i < iterations; i++)  {
      remap(src, tmp1, growMapx, growMapy, CV_INTER_LINEAR); // enlarge
      remap(src, tmp2, shrinkMapx, shrinkMapy, CV_INTER_LINEAR); // shrink
      addWeighted(tmp1, 0.5, tmp2, 0.5, 0, src); // blend back to src
    }
    imshow("temp1", tmp1);
    imshow("temp2", tmp2);
    imshow("image", src);
    imwrite("radialBlurimage.jpg", src);
    waitKey(0);
    return 1;
}
