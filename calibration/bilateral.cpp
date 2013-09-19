#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    Mat disp, newdisp;

    disp = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    int d;
    double sigmaC, sigmaS;

    d = atoi(argv[2]);
    sigmaC = atoi(argv[3]);
    sigmaS = atoi(argv[4]);

    bilateralFilter(disp, newdisp, d, sigmaC, sigmaS, BORDER_DEFAULT);

    imshow("disp", disp);
    imshow("new disp", newdisp);
    imwrite(argv[5], newdisp);

    waitKey(0);

    return(0);
}