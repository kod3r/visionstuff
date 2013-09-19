#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <dirent.h>

using namespace cv;
using namespace std;

int initalizeMat(char* fname, vector<Mat>& imgs, vector<Mat>& gray1, vector<Mat>& gray2);

int main(int argc, char* argv[])
{
    int numBoards = atoi(argv[1]);
    int board_w = atoi(argv[2]);
    int board_h = atoi(argv[3]);

    Size board_sz = Size(board_w, board_h);
    int board_n = board_w*board_h;

    vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > imagePoints1, imagePoints2;
    vector<Point2f> corners1, corners2;

    vector<Point3f> obj;
    for (int j=0; j<board_n; j++)
    {
        obj.push_back(Point3f(j/board_w, j%board_w, 0.0f));
    }

    vector<Mat> imgs;
    vector<Mat> gray1, gray2;

    int success = 0, k = 0;
    bool found1 = false, found2 = false;

    initalizeMat(argv[4], imgs, gray1, gray2);

    for(int i=0; i<imgs.size(); i++)
    {
        imshow("img1", gray1[i]);
        printf("%d\n", i);
        waitKey(100);
    }

    Mat g1, g2;
    for(int i=0; i<imgs.size(); i++)
    {
        g1 = gray1[i];
        g2 = gray2[i];

        printf("processing image pair %d\n", i);
        printf("cols = %d, rows = %d\n", g1.cols, g1.rows);

        //imshow("img1", g1);
        //waitKey(0);
        found1 = findChessboardCorners(g1, board_sz, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        found2 = findChessboardCorners(g2, board_sz, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        printf("%d %d\n", found1, found2);

        if (found1)
        {
            cornerSubPix(g1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(g1, board_sz, corners1, found1);
        }

        if (found2)
        {
            cornerSubPix(g2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(g2, board_sz, corners2, found2);
        }
        
        //imshow("image1", g1);
        //imshow("image2", g2);

        //k = waitKey(10);
        if (found1 && found2)
        {
            printf("corners found for image pair %d\n", i);
            imagePoints1.push_back(corners1);
            imagePoints2.push_back(corners2);
            object_points.push_back(obj);
            printf ("Corners stored\n");
            success++;
            //k = waitKey(0);
        }
        if (k == 27)
        {
            break;
        }
        /*
        if (k == ' ' && found1 !=0 && found2 != 0)
        {
            imagePoints1.push_back(corners1);
            imagePoints2.push_back(corners2);
            object_points.push_back(obj);
            printf ("Corners stored\n");
            success++;

            if (success >= numBoards)
            {
                break;
            }
        }*/
    }

    destroyAllWindows();
    printf("Starting Calibration\n");

    Mat CM1 = Mat(3, 3, CV_64FC1);
    Mat CM2 = Mat(3, 3, CV_64FC1);
    Mat D1, D2;
    Mat R, T, E, F;

    stereoCalibrate(object_points, imagePoints1, imagePoints2, 
                    CM1, D1, CM2, D2, g1.size(), R, T, E, F,
                    cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5), 
                    CV_CALIB_SAME_FOCAL_LENGTH + CV_CALIB_ZERO_TANGENT_DIST);

    FileStorage fs1("/home/jay/Tesseract/Evo3D/Calibration/mystereocalib14.yml", FileStorage::WRITE);
    fs1 << "CM1" << CM1;
    fs1 << "CM2" << CM2;
    fs1 << "D1" << D1;
    fs1 << "D2" << D2;
    fs1 << "R" << R;
    fs1 << "T" << T;
    fs1 << "E" << E;
    fs1 << "F" << F;

    printf("Done Calibration\n");

    printf("Starting Rectification\n");

    Mat R1, R2, P1, P2, Q;
    stereoRectify(CM1, D1, CM2, D2, g1.size(), R, T, R1, R2, P1, P2, Q);
    fs1 << "R1" << R1;
    fs1 << "R2" << R2;
    fs1 << "P1" << P1;
    fs1 << "P2" << P2;
    fs1 << "Q" << Q;

    printf("Done Rectification\n");

    printf("Applying Undistort\n");

    Mat map1x, map1y, map2x, map2y;
    Mat imgU1, imgU2;

    initUndistortRectifyMap(CM1, D1, R1, P1, g1.size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(CM2, D2, R2, P2, g1.size(), CV_32FC1, map2x, map2y);

    printf("Undistort complete\n");

    for(int i=0; i<gray1.size(); i++)
    {
        g1 = gray1[i];
        g2 = gray2[i];
        //g1 = imread("l1.pgm", CV_LOAD_IMAGE_GRAYSCALE);
        //g1 = imread("r1.pgm", CV_LOAD_IMAGE_GRAYSCALE);

        remap(g1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(g2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

        imwrite("gray1.pgm", imgU1);
        imwrite("gray2.pgm", imgU2);
    }


    return 0;
}

int initalizeMat(char* fname, vector<Mat>& imgs, vector<Mat>& gray1, vector<Mat>& gray2)
{
    printf("initalizeMat\n");
    Mat readImg;

    DIR *dpdf;
    struct dirent *epdf;

    dpdf = opendir("/home/jay/Tesseract/Evo3D/Calibration/images13/");
    if (dpdf != NULL)
    {
        while(epdf = readdir(dpdf))
        {
            printf("%s\n", epdf->d_name);
            char filename[40];
            strcpy(filename, "images13/");
            strcat(filename, epdf->d_name);
            readImg = imread(filename);

            if (readImg.data == NULL)
                continue;
            imgs.push_back(readImg);
            imshow("img", readImg);
            waitKey(100);
        }
    }
    printf("number of images loaded = %ld\n", imgs.size());
    Mat img;
    
    for(int i=0; i<imgs.size(); i++)
    {
        Mat g1, g2;
        img = imgs[i];
        Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
        Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));
        cvtColor(img1, g1, CV_BGR2GRAY);
        cvtColor(img2, g2, CV_BGR2GRAY);

        gray1.push_back(g1);
        gray2.push_back(g2);
    }
}
