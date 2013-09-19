#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <string.h>
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    Mat img, g1, g2;
    Mat disp, disp8;
    char* method = argv[2];
    img = imread(argv[1]);
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));
    printf("%d %d\n", img2.cols, img2.rows);
    //img2 = imread(argv[2]);
    //resize(img1, img1, Size(img1.cols/2, img1.rows/2));
    //resize(img2, img2, Size(img2.cols/2, img2.rows/2));
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);

    if (!(strcmp(method, "BM")))
    {
        StereoBM sbm;
        sbm.state->SADWindowSize = 9;
        sbm.state->numberOfDisparities = 112;
        sbm.state->preFilterSize = 5;
        sbm.state->preFilterCap = 61;
        sbm.state->minDisparity = 0;
        sbm.state->textureThreshold = 507;
        sbm.state->uniquenessRatio = 0;
        sbm.state->speckleWindowSize = 0;
        sbm.state->speckleRange = 8;
        sbm.state->disp12MaxDiff = 1;
        sbm(g1, g2, disp);
    }
    else if (!(strcmp(method, "SGBM")))
    {
        StereoSGBM sbm;
        sbm.SADWindowSize = 5;
        sbm.numberOfDisparities = 192;
        sbm.preFilterCap = 4;
        sbm.minDisparity = -64;
        sbm.uniquenessRatio = 1;
        sbm.speckleWindowSize = 150;
        sbm.speckleRange = 2;
        sbm.disp12MaxDiff = 10;
        sbm.fullDP = false;
        sbm.P1 = 600;
        sbm.P2 = 2400;
        sbm(g1, g2, disp);
    }

    
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

    imshow("left", img1);
    imshow("right", img2);
    imshow("disp", disp8);
    //imwrite("disp.jpg", disp8);

    double min_intensity, max_intensity;
    minMaxLoc(disp8, &min_intensity, &max_intensity);
    int i, j, sum = 0, count = 0, val;
    for(i = 0; i < disp8.cols; i++)
    {
        for(j = 0 ; j < disp8.rows; j++)
        {
            val = disp8.at<uchar>(i, j);
            if (val != 0)
            {
                    sum += disp8.at<uchar>(i, j);    
                    count++;
            }
            
        }
    }
        
    int ave;
    ave = sum/count;
    printf("%d %d %d %d\n", img1.cols, img1.rows, disp8.rows, disp8.cols);
    Mat img_thresh;
    printf("%d\n", ave);
    threshold(disp8, img_thresh, ave+10, 255,THRESH_BINARY);
    medianBlur(img_thresh, img_thresh, 9);
    Mat kernel;
    int size=2;
    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));
    //dilate(img_thresh, img_thresh, kernel);
    Mat bitwise;
    vector<Mat> channels;
    // remeber bitwise disp8 and image \m/
    channels.push_back(img_thresh);
    channels.push_back(img_thresh);
    channels.push_back(img_thresh);

    merge(channels, img_thresh);
    bitwise_and(img1, img_thresh, bitwise);
    imshow("bitwise", bitwise);
    //img_thresh = 255 - img_thresh;
    imshow("threshold", img_thresh);

    waitKey(0);

    return(0);
}
