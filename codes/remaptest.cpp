#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <math.h>
 
using namespace cv;
 
// globals
Mat src, dst;
Mat map_x, map_y;
#define REMAP_WINDOW "Remap Circle"
 
void make_circle_map(void);
 
int main(int argc, char** argv) {
        // load image
        src = imread(argv[1], 1);
 
        // create destination and the maps
        dst.create(src.size(), src.type());
        map_x.create(src.size(), CV_32FC1);
        map_y.create(src.size(), CV_32FC1);
 
        // create window
        namedWindow(REMAP_WINDOW, CV_WINDOW_AUTOSIZE);
 
        make_circle_map();
        remap(src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0));
        imshow(REMAP_WINDOW, dst);
 
        while(27 != waitKey()) {
                // just wait
        }
        cvDestroyWindow(REMAP_WINDOW);
        return 0;
}
 
void make_circle_map(void) {
        double rad = (src.rows < src.cols ? src.rows : src.cols)/2;
        double diag_rad = sqrt(src.rows*src.rows + src.cols*src.cols)/2;
        printf("radius = %d (rows: %d, cols: %d)\n", (int)rad, src.rows, src.cols);
 
        // the center 
        double c_x = (double)src.cols/2;
        double c_y = (double)src.rows/2;
 
        for(int j = 0; j < src.rows; j++) {
                for(int i = 0; i < src.cols; i++) {
                        // shift the coordinates space to center
                        double x = i-c_x;
                        double y = j-c_y;
 
                        // handle the 0 and pi/2 angles separately as we are doing atan
                        if(0 == x) {
                                double ratio = 2*rad/src.rows;
                                map_y.at<float>(j,i) = y/ratio + c_y;
                                map_x.at<float>(j,i) = c_x;
                        }
                        else if(0 == y) {
                                double ratio = 2*rad/src.cols;
                                map_x.at<float>(j,i) = x/ratio + c_x;
                                map_y.at<float>(j,i) = c_y;
                        }
                        else {
                                // get the r and theta
                                double r = sqrt(y*y + x*x);
                                double theta = atan(y/x);
                                // get the length of line at theta touching the rectangle border
                                double diag = min(fabs(c_x/cos(theta)), fabs(c_y/sin(theta)));
 
                                // scale r
                                double ratio = rad/diag;
                                r = r/ratio;
 
                                // remap the point
                                if(x > 0)       map_x.at<float>(j,i) = r*cos(fabs(theta)) + c_x;
                                else            map_x.at<float>(j,i) = c_x - r*cos(fabs(theta));
 
                                if(y > 0)       map_y.at<float>(j,i) = r*sin(fabs(theta)) + c_y;
                                else            map_y.at<float>(j,i) = c_y - r*sin(fabs(theta));
                        }
                }
        }
}