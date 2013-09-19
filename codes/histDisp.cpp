#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/**
 * @function main
 */
int main( int argc, char** argv )
{
    Mat src, dst;

  /// Load image
    src = imread( argv[1], 1 );

    cvtColor(src, src, CV_BGR2GRAY);

    if( !src.data )
        { return -1; }

    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    bool uniform = true; bool accumulate = false;

    Mat b_hist;

    calcHist(&src, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    //b_hist.at<uchar>(0, 0) = 0;
    for (int i=0; i<b_hist.rows; i++)
    {
        printf("%d - %d\n", i, b_hist.at<uchar>(0, i));
    }
    printf("%d %d\n", b_hist.rows, b_hist.cols);
    double minVal, maxVal;
    Point minLoc, maxLoc;

    minMaxLoc(b_hist, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    printf("%lf %lf\n", minVal, maxVal);
    printf("(%d, %d), (%d, %d) \n", minLoc.x, minLoc.y, maxLoc.x, maxLoc.y);

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w/histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for(int i=1; i<histSize; i++)
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
    }

    imshow("hist", histImage);
    imshow("disp", src);
    waitKey(0);
    return 0;
}