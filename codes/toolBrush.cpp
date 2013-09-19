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
bool pointCdraw;

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    point1 = Point(x, y);
    pointC = Point(x, y);
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        /* left button clicked. ROI selection begins */
        
        pointCdraw = !pointCdraw;
        //printf("points %d %d\n", x, y);
    }
}

int main(int argc, char* argv[])
{
    point1 = Point(0 ,0);
    pointC = Point(0, 0);
    Mat img, displayImg;
    int rad;
    int k;
    img = imread(argv[1]);
    rad = atoi(argv[2]);
    img.copyTo(displayImg);
    imshow("display", displayImg);
    cvSetMouseCallback("display", mouseHandler, NULL);
    while(true)
    {
        //pointCdraw = false;
        img.copyTo(displayImg);
        circle(displayImg, point1, rad, Scalar(255, 255, 255), -1);
        imshow("display", displayImg);
        k = waitKey(1);
        if (pointCdraw)
        {
            circle(img, pointC, rad, Scalar(255, 255, 255), -1);
        }
        if (k == 27)
        {
            break;
        }

    }
    return(0);
}