#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
using namespace std;
using namespace cv;
bool apply(Mat& img);
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
    imshow("prev", src);
    apply(src);
    imshow("after", src);
    waitKey(0);
    return 1;
}
bool apply(Mat& img) {
   int dim(256);

   Mat lut(1, &dim, CV_8UC(img.channels()));

   if( img.channels() == 1)
   {
      for (int i=0; i<256; i++)
         lut.at<uchar>(i)= 255-i;
   }
   else // stupid idea that all the images are either mono either multichannel
   {
      for (int i=0; i<256; i++)
      {
         lut.at<Vec3b>(i)[0]= 255-i;   // first channel  (B)
         lut.at<Vec3b>(i)[1]= 255-i/2; // second channel (G)
         lut.at<Vec3b>(i)[2]= 255-i/3; // ...            (R)
      }
   }

   LUT(img,lut,img); // are you sure you are doing final->final? 
   // if yes, correct the LUT allocation part

   return true;
}