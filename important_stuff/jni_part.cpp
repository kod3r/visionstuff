#include <jni.h>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/photo/photo.hpp"
#include <stdio.h>
#include <vector>
#include <stdio.h>
#include <android/log.h>
#include <cstdlib>
#include <string>

#define APPNAME "Studio 3d jni"

#define LOGD(TAG) __android_log_print(ANDROID_LOG_DEBUG , APPNAME,TAG);


using namespace std;
using namespace cv;

int getDisp(Mat g1, Mat g2, Mat &disp);
int getThreshold(Mat img, Point p1, int range, Mat &foreground);
int segmentForeground(Mat img, Mat &foreground, Mat &background,vector <vector<Point> > &contours);
int getBlurMaskedImage(Mat img, Mat &foreground);
int getMaskedImage(Mat img, Mat &foreground);
int addFgBg(Mat foreground, Mat background, Mat &img);
int getBlurMaskedGrayImage(Mat img, Mat &foreground);
int getMaskedGrayImage(Mat img, Mat &foreground);
int deFocus(Mat img, Mat& foreground, int size, int radius);
int deFocus8(Mat img, Mat& foreground, int size, int w, int h);
int doBokeh(Mat disp, Mat img);
int doBokehImg(Mat disp, Mat img, Mat& foreground);
int getSepia(Mat img, Mat &foreground);
int doBokehImgRelative(Mat disp, Mat img, Mat& foreground, Point p1);
int getGaussianBlur(Mat img, Mat& retVal, int ksize);
int getThresh(Mat img, Mat& retVal, int l1, int l2, int h1, int h2);
int stackUp(vector<Mat>& layers, Mat& retVal);
int doMultiBlur(Mat img, Mat& retVal, Mat disp, Point p1);
int doCircBlur(Mat img, Mat& retVal, int radius);
int getDisparity(Mat g1, Mat g2, Mat &disp);
int segmentBlurs(Mat img, Mat &foreground);
int doOilPaint(Mat src, Mat& foreground);
int doGraySingle(Mat img, Mat& retVal, Mat disp, Point p1);
int getRange(Mat disp, Point p1);
int stickImage(Mat &foreground, Mat &background);
int histPick(Mat disp);
int getThresholdHist(Mat img, int dispval, int range, Mat &foreground);
int doMultiBlurHist(Mat img, Mat& retVal, Mat disp, int dispval);

// New functions for automatic picking
int histPick(Mat Oirgdisp, int &lower, int &upper);
int histRange(int pxv[], int index, int& lower, int& upper);
int histSegmentForeground(Mat &foreground, Mat &background);

static jfloatArray gArray = NULL;
static int width,height;

int getGray(Mat& img)
{
  cvtColor(img, img, CV_BGR2GRAY);
  return 1;
}
extern "C" {
JNIEXPORT jfloatArray JNICALL Java_com_tesseract_studio3d_Animation_PhotoActivity_getThreshold(JNIEnv* env, jobject, jlong addrBgr, jlong addrDisp, jlong addrBackground,jlong addrForeground,jlong finalImage, jint ji1, jint ji2,jint currentMode);
JNIEXPORT void JNICALL Java_com_tesseract_studio3d_Animation_AnimationActivity_getThreshold(JNIEnv* env, jobject, jlong addrBgr, jlong addrDisp, jlong addrBackground,jlong addrForeground,jlong finalImage, jint ji1, jint ji2,jint currentMode);
JNIEXPORT void JNICALL Java_com_tesseract_studio3d_Animation_PhotoActivity_getDisparity(JNIEnv*, jobject, jlong addrRgba, jlong finalImage);
JNIEXPORT void JNICALL Java_com_tesseract_studio3d_Animation_PhotoActivity_crop5(JNIEnv*, jobject, jlong addrRgba, jlong finalImage);
JNIEXPORT void JNICALL Java_com_tesseract_studio3d_selectionscreen_MainScreen_getDisparity(JNIEnv*, jobject, jlong addrRgba, jlong finalImage);

JNIEXPORT void JNICALL Java_com_tesseract_studio3d_Animation_AnimationActivity_reFocus(JNIEnv* env, jobject, jlong addrBgr, jlong addrDisp,jlong finalImage, jint ji1, jint ji2);

JNIEXPORT void JNICALL Java_com_tesseract_studio3d_refocus_FocusImageView_Refocus(JNIEnv* env, jobject, jlong addrBgr, jlong addrDisp,jlong finalImage, jint ji1, jint ji2);

//JNIEXPORT void JNICALL Java_com_tesseract_studio3d_replace_ReplaceActivity_getThreshold(JNIEnv* env, jobject, jlong addrBgr, jlong addrDisp, jlong finalImage,jlong addrBackground,jlong addrForeground, jint ji1, jint ji2,jint currentMode,jstring imgPath);
JNIEXPORT void JNICALL Java_com_tesseract_studio3d_replace_ReplaceActivity_getThreshold(JNIEnv* env, jobject, jlong addrBgr, jlong addrDisp, jlong finalImage,jlong addrBackground,jlong addrForeground, jint ji1, jint ji2,jint currentMode,jlong loadedImgMat);
JNIEXPORT void JNICALL Java_com_tesseract_studio3d_manualEdit_Panel_updateDisp(JNIEnv* env, jobject);

JNIEXPORT void JNICALL Java_com_tesseract_studio3d_refocus_FocusImageView_Refocus(JNIEnv* env, jobject, jlong addrBgr, jlong addrDisp,jlong finalImage, jint ji1, jint ji2)
{
    /* JNI Part for Refcous
     * Takes in Disparity, left image and touch point.
     * Segments layers based on depth range, applies gaussian blur
     * Stacks all layers and returns the image
     */


  Mat& img = *(Mat*)addrBgr;
  Mat& disp = *(Mat*)addrDisp;

  Mat background;
  Mat foreground;

  Mat& finImg = *(Mat*)finalImage;
  LOGD("Initialize");

  vector<vector<Point> > contours;

  Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
  Point point1;

  int x, y;
    x = ji1;
    y = ji2;

    point1 = Point(x, y); // to get from android
    LOGD("Point initial");


    getThreshold(disp, point1, 10, foreground);
    LOGD("THREESH");
    segmentForeground(img1, foreground, background, contours);
    int tLen=0;
  char str[10];
  char str2[]={" Value"};
  sprintf(str, "%d", tLen);
  strcat(str,str2);

  LOGD (str);
    char str3[]={"dimensions"};

    sprintf(str, "%d", img1.rows);
    strcat(str,str3);
    LOGD(str);

    sprintf(str, "%d", img1.cols);
    strcat(str,str3);
    LOGD(str);

    sprintf(str, "%d", disp.rows);
    strcat(str,str3);
    LOGD(str);

    sprintf(str, "%d", disp.cols);
    strcat(str,str3);
    LOGD(str);


    Mat blurBackground;
    // get BlurBackground
    doMultiBlur(img1, blurBackground, disp, point1);
    bitwise_and(background, blurBackground, background);
    LOGD("Reached the end");

    // get foreground
    getMaskedImage(img1, foreground);

    imwrite("/mnt/sdcard/Studio3D/img_refocus_fg22.png", foreground);
    imwrite("/mnt/sdcard/Studio3D/img_refocus_bg22.png", background);

    // add fg and bg to get final image
    addFgBg(foreground, background, finImg);
    cvtColor(finImg, finImg,CV_BGR2RGBA);

    imwrite("/mnt/sdcard/Studio3D/img_refocus_finImg.png", background);
}


JNIEXPORT void JNICALL Java_com_tesseract_studio3d_Animation_PhotoActivity_crop5(JNIEnv*, jobject, jlong addrRgba, jlong finalImage)
{
  Mat& img = *(Mat*)addrRgba;
  Mat& retVal = *(Mat*)finalImage;

  Mat img1(img, Rect(70, 100, 500, 500));
  img1.copyTo(retVal);

}



JNIEXPORT void JNICALL Java_com_tesseract_studio3d_Animation_AnimationActivity_getThreshold(JNIEnv* env, jobject, jlong addrBgr, jlong addrDisp, jlong finalImage,jlong addrBackground,jlong addrForeground, jint ji1, jint ji2,jint currentMode)
{

    // not used anymore

  LOGD("Start");
  Mat& img = *(Mat*)addrBgr;
  img = imread("/mnt/sdcard/Studio3D/img_full.jpg");
  Mat& disp = *(Mat*)addrDisp;
  disp = imread("/mnt/sdcard/Studio3D/disp.png");
  cvtColor(disp, disp, CV_BGR2GRAY);
  Mat& background = *(Mat*)addrBackground;
  Mat& foreground = *(Mat*)addrForeground;

  Mat& finImg = *(Mat*)finalImage;
  finImg = Mat::zeros(finImg.rows, finImg.cols, CV_8UC3);
  LOGD("Initialize");

  jfloatArray contourPoints;
  vector<vector<Point> > contours;

  Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
  Point point1;

    int x, y;
    x = ji1;
    y = ji2;

    point1 = Point(x, y); // to get from android
    LOGD("Point initial");


    getThreshold(disp, point1, 15, foreground);
    LOGD("THREESH");
    segmentForeground(img1, foreground, background,contours);


    LOGD ("Another breakpt");
    Mat layerAf, layerAb;
    cvtColor(foreground, layerAf, CV_BGR2GRAY);
    cvtColor(background, layerAb, CV_BGR2GRAY);

    LOGD ("Segmented");

    int tLen=0;

  for(int i=0; i<contours.size(); i++)
  {
    for(int j=0; j<contours[i].size(); j++)
    {
      tLen++;
    }
  }
//LOGD("Size"+tLen);

     LOGD ("Start contour points");
  contourPoints = env->NewFloatArray(2*tLen);
  jfloat cPoints[2*tLen];

  char str[10];
  char str2[]={" Value"};
  sprintf(str, "%d", tLen);
  strcat(str,str2);

   LOGD (str);

  tLen=0;
  for(int i=0; i<contours.size(); i++)
  {
    for(int j=0; j<contours[i].size(); j++)
    {
      cPoints[tLen] = contours[i][j].x;
      tLen++;
      cPoints[tLen] = contours[i][j].y;
      tLen++;
    }
  }

     LOGD ("Completed loop");

     char str3[]={"dimensions"};

       sprintf(str, "%d", img1.rows);
       strcat(str,str3);

       imwrite("/mnt/sdcard/Studio3D/img_jni_img.png", finImg);
//       sprintf(str3, "%d  ", img1.cols);
//       strcat(str,"  ");
//       strcat(str,str3);

       LOGD(str);

    if(currentMode==1)
    {
    Mat blurBackground;
        doMultiBlur(img1, blurBackground, disp, point1);
        bitwise_and(background, blurBackground, background);
  }
    else if(currentMode==2)
    doOilPaint(img1, background);
    else if(currentMode==3)
    getMaskedGrayImage(img1, background);
    else if(currentMode==4)
    getSepia(img1, background);
    else if(currentMode == -1)
        getMaskedImage(img1, background);

    LOGD("Reached the end");
    getMaskedImage(img1, foreground);

    imwrite("/mnt/sdcard/Studio3D/img_fg12_mid.png", foreground);
    imwrite("/mnt/sdcard/Studio3D/img_bg12_mid.png", background);

    cvtColor(foreground, foreground, CV_BGR2RGBA);
    cvtColor(background, background, CV_BGR2RGBA);

    vector<Mat> rgbam;
    split(foreground, rgbam);
    rgbam[3] = layerAf;
    merge(rgbam, foreground);

    rgbam.clear();

    split(background, rgbam);
    rgbam[3] = layerAb;
    merge(rgbam, background);
    rgbam.clear();

   // imwrite("/mnt/sdcard/Studio3D/layers/img_fg22.png", foreground);
    //imwrite("/mnt/sdcard/Studio3D/layers/img_bg22.png", background);

    addFgBg(foreground, background, finImg);
    imwrite("/mnt/sdcard/Studio3D/img_fin22.png", finImg);
   // imwrite("/mnt/sdcard/Studio3D/layers/img_fin.png", finImg);
    //resize(finImg, finImg, Size(finImg.cols*2, finImg.rows));


//    env->SetFloatArrayRegion(contourPoints, 0, tLen, cPoints);

   // jsize arraylen=env->GetArrayLength( contourPoints);
  //  sprintf(str, "%d", arraylen);

    LOGD(str);



}


JNIEXPORT void JNICALL Java_com_tesseract_studio3d_replace_ReplaceActivity_getThreshold(JNIEnv* env, jobject, jlong addrBgr, jlong addrDisp, jlong finalImage,jlong addrBackground,jlong addrForeground, jint ji1, jint ji2,jint currentMode,jlong addrLoadedImage)
{

    /* This function supports Replace3D Activity
     * left image and disparity map are supposed to be passed, but are
     * read using imread. Needs to be changed.
     * cuurentMode = 6 always for this function. Also, need to clean up
     * 
     */

//  const char *cparam = env->GetStringUTFChars(path, 0);
//  string imgPath=cparam;
	LOGD("JNI startttt");



  LOGD("Start");
  Mat& img = *(Mat*)addrBgr;
  img = imread("/mnt/sdcard/Studio3D/img_full.jpg");
  Mat& disp = *(Mat*)addrDisp;
  disp = imread("/mnt/sdcard/Studio3D/disp.png");
  cvtColor(disp, disp, CV_BGR2GRAY);
  Mat& background = *(Mat*)addrBackground;
  Mat& foreground = *(Mat*)addrForeground;



  Mat& finImg = *(Mat*)finalImage;
  finImg = Mat::zeros(finImg.rows, finImg.cols, CV_8UC3);
  LOGD("Initialize");

  jfloatArray contourPoints;
  vector<vector<Point> > contours;
LOGD("INITIALIZE RECT IMG");
  Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
  Point point1;
//
    int x, y;
    x = ji1;
    y = ji2;

    int dispval;
	int lower, upper;
    LOGD("HIST PICK");
    // Pick the most suitable disparity value based on histogram of
    // the disparity image. Also get upper and lower range
    dispval = histPick(disp, lower, upper);
    LOGD("get thresh");
    // threshold disparity image
    inRange(disp, Scalar(lower), Scalar(upper), foreground);
    LOGD("inRange done");
    // segment foreground and baclground layers. Returned images are 3 layers
    histSegmentForeground(foreground, background);
    imwrite("/mnt/sdcard/Studio3D/replace_fg_mask_int.png", foreground);
    imwrite("/mnt/sdcard/Studio3D/replace_bg_mask_int.png", foreground);
    LOGD("segmentation done");
    //getThresholdHist(disp, dispval, 15, foreground);


    //point1 = Point(x, y); // to get from android
    //LOGD("Point initial");


    //getThreshold(disp, point1, 10, foreground);
    LOGD("THREESH");
    //segmentForeground(img1, foreground, background,contours);


    LOGD ("Another breakpt");

    // Saving background and foreground single layers to be used as
    // Alpha layers later in the function
    Mat layerAf, layerAb;
    cvtColor(foreground, layerAf, CV_BGR2GRAY);
    cvtColor(background, layerAb, CV_BGR2GRAY);

    LOGD ("Segmented");

    int tLen=0;

    // This contour stuff is bullshit. Needs to be cleaned up.
  for(int i=0; i<contours.size(); i++)
  {
    for(int j=0; j<contours[i].size(); j++)
    {
      tLen++;
    }
  }
//LOGD("Size"+tLen);

     LOGD ("Start contour points");
  contourPoints = env->NewFloatArray(2*tLen);
  jfloat cPoints[2*tLen];

  char str[10];
  char str2[]={" Value"};
  sprintf(str, "%d", tLen);
  strcat(str,str2);

   LOGD (str);

  tLen=0;
  for(int i=0; i<contours.size(); i++)
  {
    for(int j=0; j<contours[i].size(); j++)
    {
      cPoints[tLen] = contours[i][j].x;
      tLen++;
      cPoints[tLen] = contours[i][j].y;
      tLen++;
    }
  }

     LOGD ("Completed loop");

     char str3[]={"dimensions"};

       sprintf(str, "%d", img1.rows);
       strcat(str,str3);

       imwrite("/mnt/sdcard/Studio3D/img_jni_img.png", finImg);
//       sprintf(str3, "%d  ", img1.cols);
//       strcat(str,"  ");
//       strcat(str,str3);

       LOGD(str);

       if(currentMode==1)
         {
            // useless
             Mat blurBackground;
             doMultiBlur(img1, blurBackground, disp, point1);
             bitwise_and(background, blurBackground, background);
             getMaskedImage(img1, foreground);
         }
         else if(currentMode==2)
         {
            // useless
             doOilPaint(img1, background);
             getMaskedImage(img1, foreground);
         }
         else if(currentMode==3)
         {
            // useless
             getMaskedGrayImage(img1, background);
             getMaskedImage(img1, foreground);
         }
         else if(currentMode==4)
         {
            // useless
             getSepia(img1, background);
             getMaskedImage(img1, foreground);
         }
         else if(currentMode == 5)
         {
            // useless
             Mat stickimg;
             //stickimg = imread(imgPath);
             resize(stickimg, stickimg, Size(background.cols, background.rows));
             bitwise_and(background, stickimg, stickimg);
             stickimg.copyTo(background);
             getMaskedImage(img1, foreground);
         }
         else if (currentMode == 6)
         {
            // Mat stickimg;
           //  imgPath="/mnt/sdcard/download/bungee/A1.jpg";
            // LOGD(cparam);

             //stickimg = imread(imgPath);

             Mat& stickimg = *(Mat*)addrLoadedImage;

             sprintf(str, "%d", stickimg.rows);
             strcat(str,str2);

             LOGD(str);

             // resize image if they have different sizes
             resize(stickimg, stickimg, Size(foreground.cols, foreground.rows));

             // put wallpaper in the background
             bitwise_and(background, stickimg, stickimg);
             stickimg.copyTo(background);

             // put image in the foregorund
             getMaskedImage(img1, foreground);

             //resize(stickimg, stickimg, Size(foreground.cols, foreground.rows));
             //bitwise_and(foreground, stickimg, stickimg);
             //stickimg.copyTo(foreground);
             //getMaskedImage(img1, background);
         }
         else if(currentMode == -1)
         {
             getMaskedImage(img1, background);
             getMaskedImage(img1, foreground);
       }

    LOGD("Reached the end");
//    getMaskedImage(img1, foreground);

    imwrite("/mnt/sdcard/Studio3D/img_fg12_mid.png", foreground);
    imwrite("/mnt/sdcard/Studio3D/img_bg12_mid.png", background);

    // Using RGBA so that alpha channel can be used effectively
    cvtColor(foreground, foreground, CV_BGR2RGBA);
    cvtColor(background, background, CV_BGR2RGBA);

    vector<Mat> rgbam;
    split(foreground, rgbam);

    // change alpha layer to the one that was saved previously
    rgbam[3] = layerAf;
    merge(rgbam, foreground);
    rgbam.clear();

    split(background, rgbam);
    // change alpha layer to the one that was saved previously
    rgbam[3] = layerAb;
    merge(rgbam, background);
    rgbam.clear();

   // imwrite("/mnt/sdcard/Studio3D/layers/img_fg22.png", foreground);
    //imwrite("/mnt/sdcard/Studio3D/layers/img_bg22.png", background);

    addFgBg(foreground, background, finImg);
    cvtColor(finImg, finImg,CV_BGR2RGBA);
    imwrite("/mnt/sdcard/Studio3D/img_fin22.png", finImg);
   // imwrite("/mnt/sdcard/Studio3D/layers/img_fin.png", finImg);
    //resize(finImg, finImg, Size(finImg.cols*2, finImg.rows));


//    env->SetFloatArrayRegion(contourPoints, 0, tLen, cPoints);

   // jsize arraylen=env->GetArrayLength( contourPoints);
  //  sprintf(str, "%d", arraylen);

    LOGD(str);

	// .. do something with it
 // env->ReleaseStringUTFChars(path, cparam);


}






JNIEXPORT void JNICALL Java_com_tesseract_studio3d_Animation_PhotoActivity_getDisparity(JNIEnv*, jobject, jlong addrRgba, jlong finalImage)
{

    /* This function returns and saves Disparity image. */
    Mat& img = *(Mat*)addrRgba;
    Mat g1, g2;
    Mat& disp = *(Mat*)finalImage;
    cvtColor(img, img, CV_RGBA2BGR);
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    getDisp(g1, g2, disp);
    imwrite("/mnt/sdcard/Studio3D/disp.png", disp);

    return;
}

JNIEXPORT void JNICALL Java_com_tesseract_studio3d_selectionscreen_MainScreen_getDisparity(JNIEnv*, jobject, jlong addrRgba, jlong finalImage)
{
    /* This function returns and saves Disparity image. */
    Mat& img = *(Mat*)addrRgba;
    Mat g1, g2;
    Mat& disp = *(Mat*)finalImage;
    cvtColor(img, img, CV_RGBA2BGR);
    Mat img1(img, Rect(0, 0, img.cols/2, img.rows));
    Mat img2(img, Rect(img.cols/2, 0, img.cols/2, img.rows));
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    getDisp(g1, g2, disp);
    imwrite("/mnt/sdcard/Studio3D/disp.png", disp);

    return;
}

JNIEXPORT jfloatArray JNICALL Java_com_tesseract_studio3d_Animation_PhotoActivity_getThreshold(JNIEnv* env, jobject, jlong addrBgr, jlong addrDisp, jlong finalImage,jlong addrBackground,jlong addrForeground, jint ji1, jint ji2,jint currentMode)
{
    /* This functions supports Photo3D application
     * Object nearest in the scene is segmented automatically
     * Two layers are made, foreground and background. saved and returned.
     */

  String path;
  Mat& img = *(Mat*)addrBgr;
  Mat& disp = *(Mat*)addrDisp;

  Mat& background = *(Mat*)addrBackground;
  Mat& foreground = *(Mat*)addrForeground;

  Mat& finImg = *(Mat*)finalImage;

  jfloatArray contourPoints;
  vector<vector<Point> > contours;

  Mat img1(img, Rect(0, 0, img.cols/2, img.rows));


    char str3[10];
    char str4[]={"Size   "};
    sprintf(str3, "%d", img1.cols);
    strcat(str4,str3);
    LOGD(str4);

  Point point1;

    // no need now
    int x, y;
    x = ji1;
    y = ji2;

    point1 = Point(x, y); // to get from android // useless now

    //getThreshold(disp, point1, 10, foreground);

    int dispval;
    int lower, upper;

    // Pick the most suitable disparity value based on histogram of
    // the disparity image. Also get upper and lower range
    dispval = histPick(disp, lower, upper);
    // threshold disparity image
    inRange(disp, Scalar(lower), Scalar(upper), foreground);
    // segment foreground and baclground layers. Returned images are 3 layers
    histSegmentForeground(foreground, background);
    //segmentForeground(img1, foreground, background,contours);

    imwrite("/mnt/sdcard/Studio3D/img_mask_fg_old.png",foreground);
    imwrite("/mnt/sdcard/Studio3D/img_mask_bg_old.png",background);

    imwrite("/mnt/sdcard/Studio3D/img_mask_fg.png",foreground);
    imwrite("/mnt/sdcard/Studio3D/img_mask_bg.png",background);

    /* get single layer fg and bg to be used as alpha layers */
    Mat layerAf, layerAb;
    cvtColor(foreground, layerAf, CV_BGR2GRAY);
    cvtColor(background, layerAb, CV_BGR2GRAY);

    LOGD ("Segmented");
    /* useless contours. Needs to be removed */
    int tLen=0;
  for(int i=0; i<contours.size(); i++)
  {
    for(int j=0; j<contours[i].size(); j++)
    {
      tLen++;
    }
  }


     LOGD ("Start contour points");
  contourPoints = env->NewFloatArray(2*tLen);
  jfloat cPoints[2*tLen];

  char str[10];
  char str2[]={"Value"};
  sprintf(str, "%d", tLen);
  strcat(str,str2);

   LOGD (str);

  tLen=0;
  for(int i=0; i<contours.size(); i++)
  {
    for(int j=0; j<contours[i].size(); j++)
    {
      cPoints[tLen] = contours[i][j].x;
      tLen++;
      cPoints[tLen] = contours[i][j].y;
      tLen++;
    }
  }

     LOGD ("Completed loop");

    if(currentMode==1)
    {
        /* not used */
      Mat blurBackground;
        doMultiBlur(img1, blurBackground, disp, point1); // why crash ?

        LOGD("did blurring");
        bitwise_and(background, blurBackground, background);
  }
    else if(currentMode==2)
    {
        /* not used */
        doOilPaint(img1, background);
        getMaskedImage(img1, foreground);
    }
    else if(currentMode==3)
    {
        /* not used */
        getMaskedGrayImage(img1, background);
        getMaskedImage(img1, foreground);
    }
    else if(currentMode==4)
    {
        /* not used */
        getSepia(img1, background);
        getMaskedImage(img1, foreground);
    }
    else if(currentMode == 5)
    {
        /* not used */
        Mat stickimg;
        stickimg = imread(path);
        resize(stickimg, stickimg, Size(background.cols, background.rows));
        bitwise_and(background, stickimg, stickimg);
        stickimg.copyTo(background);
        getMaskedImage(img1, foreground);
    }
    else if (currentMode == 6)
    {
        /* not used */
        Mat stickimg;
        stickimg = imread(path);
        resize(stickimg, stickimg, Size(foreground.cols, foreground.rows));
        bitwise_and(foreground, stickimg, stickimg);
        stickimg.copyTo(foreground);
        getMaskedImage(img1, background);
    }
    else if(currentMode == -1)
    {
        /* used */
        getMaskedImage(img1, background);
        getMaskedImage(img1, foreground);
  }

    LOGD("Reached the end");
    getMaskedImage(img1, foreground);

    /* convert BGR to RGBA to use the alpha layer */
    cvtColor(foreground, foreground, CV_BGR2RGBA);
    cvtColor(background, background, CV_BGR2RGBA);

    vector<Mat> rgbam;
    split(foreground, rgbam);
    /* change the alpha layer to fg single layer */
    rgbam[3] = layerAf;
    merge(rgbam, foreground);
    rgbam.clear();

    split(background, rgbam);
    /* change the alpha layer to bg single layer */
    rgbam[3] = layerAb;
    merge(rgbam, background);
    rgbam.clear();

    imwrite("/mnt/sdcard/Studio3D/Layers/img_fg.png", foreground);
    imwrite("/mnt/sdcard/Studio3D/Layers/img_bg.png", background);

    addFgBg(foreground, background, finImg);
    imwrite("/mnt/sdcard/Studio3D/img_fin.png", finImg);
    //resize(finImg, finImg, Size(finImg.cols*2, finImg.rows));


    env->SetFloatArrayRegion(contourPoints, 0, tLen, cPoints);

    jsize arraylen=env->GetArrayLength( contourPoints);
    sprintf(str, "%d", arraylen);

    LOGD(str);
    return contourPoints;

}

JNIEXPORT void JNICALL Java_com_tesseract_studio3d_manualEdit_Panel_updateDisp(JNIEnv* env, jobject)
{
    /* This function supports ManualEdit application.
     * Loads old and new mask and updates disparity
     */
	Mat disp, pMask, nMask, dMask, dnMask, valMat;
    disp =  imread("/mnt/sdcard/Studio3D/disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    pMask = imread("/mnt/sdcard/Studio3D/img_mask_fg_old.png", CV_LOAD_IMAGE_GRAYSCALE);
    nMask = imread("/mnt/sdcard/Studio3D/img_mask_fg.png", CV_LOAD_IMAGE_GRAYSCALE);

    /* Here, dMask is difference between new mask and old mask
     * Things newly added to foreground will be present here
     */
    dMask = nMask - pMask;
    dnMask = Scalar(255) - dMask;
	int val, lower, upper;

    /* since there is no reference point, it computes the disparity
     * value using the same function which was used to calcluate
     * disparity value in Photo3D.
     */
	val = histPick(disp, lower, upper);
    valMat = val * Mat::ones(disp.size(), CV_8U);

    /* dMask must have values of the disparity value got from histPick */
    bitwise_and(valMat, dMask, dMask);
    /* copy disparity in the rest of the area */
    bitwise_and(disp, dnMask, dnMask);

    Mat newDisp;
    /* add dMask and dnMask to get new disparity map */
    add(dMask, dnMask, newDisp);

    /* Here, dMask is the difference between old mask and new mask
     * Things which are added in background will be present here */
    dMask = pMask - nMask;

    /* Inpaint the disparity based on dMask */
    inpaint(newDisp, dMask, newDisp, 5, INPAINT_TELEA);
    
    imwrite("/mnt/sdcard/Studio3D/disp.png", newDisp);
	
}
}


int getDisp(Mat g1, Mat g2, Mat &disp)
{
    /* get Disparity Map */
    Mat disp16;
    StereoSGBM sbm;
    sbm.SADWindowSize = 7; // 5
    sbm.numberOfDisparities = 112;
    sbm.preFilterCap = 20;
    sbm.minDisparity = -64; // -64
    sbm.uniquenessRatio = 1; // 1
    sbm.speckleWindowSize = 120; //150
    sbm.speckleRange = 2;
    sbm.disp12MaxDiff = 10; // 10
    sbm.fullDP = false;
    sbm.P1 = 600;
    sbm.P2 = 2400;
    /*
    sbm.SADWindowSize = 5; // 5
    sbm.numberOfDisparities = 112;
    sbm.preFilterCap = 61;
    sbm.minDisparity = -39; // -64
    sbm.uniquenessRatio = 1; // 1
    sbm.speckleWindowSize = 180; //150
    sbm.speckleRange = 2;
    sbm.disp12MaxDiff = 20; // 10
    sbm.fullDP = false;
    sbm.P1 = 600;
    sbm.P2 = 2400;
    */
    Mat preBilateralDisp;
    sbm(g1, g2, disp16);
    normalize(disp16, preBilateralDisp, 0, 255, CV_MINMAX, CV_8U);
    bilateralFilter(preBilateralDisp, disp, 5, 50, 0);
    if (disp.cols > 0 && disp.rows > 0)
    {
        return 1;
    }
    return 0;
}

int getThreshold(Mat img, Point p1, int range, Mat &foreground)
{
    /* threshold disparity based on range and disparity value at the given point */
    int disval;
    disval = img.at<uchar>(p1.y, p1.x);
    inRange(img, disval - range, disval + range, foreground);
    medianBlur(foreground, foreground, 9);
    return 1;
}

int segmentForeground(Mat img, Mat &foreground, Mat &background, vector<vector<Point> >& contours)
{
    /* segment foreground and background based on thresholded image
     * Try to remove small objects in the background which are detected
     * as foregorund using contour size */

    //vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat drawing, kernel;
    int size=3;
    drawing = Mat::zeros(img.size(), CV_8UC3);
    findContours(foreground.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 15000)
        {
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }

    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));
    erode(drawing, drawing, kernel, Point(-1, -1), 2);
    Mat temp;
    dilate(drawing, temp, kernel, Point(-1, -1), 1);
    drawing = Mat::zeros(img.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    cvtColor(temp, temp, CV_BGR2GRAY);
    findContours(temp.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 15000)
        {
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    foreground = drawing.clone();
    background = Scalar(255, 255, 255) - foreground; // USE THIS FOR GETTING THE INVERSE ...
    return 1;
}

int getBlurMaskedGrayImage(Mat img, Mat &foreground)
{
    /* get Blur gray image */
    Mat blur, blurGray;
    cvtColor(img, blurGray, CV_BGR2GRAY);
    GaussianBlur(blurGray, blurGray, Size(9, 9), 11, 11);
    vector<Mat> gray;
    gray.push_back(blurGray);
    gray.push_back(blurGray);
    gray.push_back(blurGray);
    merge(gray, blur);
    bitwise_and(blur, foreground, foreground);
    return 1;
}
int getBlurMaskedImage(Mat img, Mat &foreground)
{
    /* get Blur image */
    Mat blur;
    GaussianBlur(img, blur, Size(13, 13), 15, 15);
    bitwise_and(blur, foreground, foreground);
    return 1;
}
int getMaskedImage(Mat img, Mat &foreground)
{
    /* get masked image */
    bitwise_and(img, foreground, foreground);
    return 1;
}

int getMaskedGrayImage(Mat img, Mat &foreground)
{
    /* get masked gray image */
    Mat g, bitwise;
    cvtColor(img, g, CV_BGR2GRAY);
    vector<Mat> gray;
    gray.push_back(g);
    gray.push_back(g);
    gray.push_back(g);
    merge(gray, bitwise);
    bitwise_and(bitwise, foreground, foreground);
    return 1;
}

int addFgBg(Mat foreground, Mat background, Mat &img)
{
    /* add foreground and background image */
    Mat tempImg;
    add(foreground, background, img);
    return 1;
}

int deFocus(Mat img, Mat& foreground, int size, int radius)
{
    /* use circular defocusing technique */
    Mat filter, blur;
    float totalSum;
    filter = Mat::zeros(size, size, CV_64F);
    circle(filter, Point(size/2, size/2), radius, Scalar(1, 1, 1), -1);
    totalSum = sum(filter)[0];
    filter = filter/totalSum;
    filter2D(img, blur, -1, filter);
    bitwise_and(blur, foreground, foreground);
    return 1;
}

int doBokehImg(Mat disp, Mat img, Mat& foreground)
{
    /* Add bokeh blur */
    int i, j, disval, size=30, dia;
    float tSum;
    Mat cImg;
    Mat bImg, filter;
    cImg = img.clone();
    bImg = img.clone();

    for(i=0; i<img.rows-size; i+=size/2)
    {
        for(j=0; j<img.cols-size; j+=size/2)
        {
            Mat subDisp = disp(Range(i, i+size), Range(j, j+size));
            Mat subImg = cImg(Range(i, i+size), Range(j, j+size));
            disval = sum(subDisp)[0]/(size*size);
            dia = 13 - disval/20;
            if (dia < 2)
                continue;
            filter = Mat::zeros(15, 15, CV_64F);
            //printf("(%d, %d) %d %d\n", i, j, dia, disval);
            circle(filter, Point(7, 7), dia/2+1, (1, 1, 1), -1);
            tSum = sum(filter)[0];
            filter = filter/tSum;
            filter2D(subImg, subImg, -1, filter);
            //subImg.copyTo(bImg(Rect(i, j, size, size)));
            subImg.copyTo(bImg.colRange(j, j+size).rowRange(i, i+size));
            //bImg(Range(i, i+size), Range(j, j+size)) = subImg;
        }
    }
    //imshow("bokeh", bImg);
    bitwise_and(bImg, foreground, foreground);
    return 1;
}

int getSepia(Mat img, Mat &foreground)
{
    /* get sepia masked image */
    Mat cImg, kernel;
    kernel = (cv::Mat_<float>(3,3) <<  0.272, 0.534, 0.131,
                                        0.349, 0.686, 0.168,
                                        0.393, 0.769, 0.189);//,
                                        //0, 0, 0, 1);
    transform(img, cImg, kernel);
    bitwise_and(cImg, foreground, foreground);
    return 1;
}

int doBokehImgRelative(Mat disp, Mat img, Mat& foreground, Point p1)
{
    /* do bokeh blur based on depth image */
    int dispVal;
    dispVal = disp.at<uchar>(p1.y, p1.x);
    int i, j, disval, size=30, dia, diff;
    float tSum;
    Mat cImg;
    Mat bImg, filter;
    cImg = img.clone();
    bImg = img.clone();

    for(i=0; i<img.rows-size; i+=size/2)
    {
        for(j=0; j<img.cols-size; j+=size/2)
        {
            Mat subDisp = disp(Range(i, i+size), Range(j, j+size));
            Mat subImg = cImg(Range(i, i+size), Range(j, j+size));
            disval = sum(subDisp)[0]/(size*size);
            diff = abs(dispVal - disval);
            if (diff < 12)
                continue;
            dia = diff/10;
            if (dia < 2)
                continue;
            filter = Mat::zeros(15, 15, CV_64F);
            //printf("(%d, %d) %d %d\n", i, j, dia, disval);
            circle(filter, Point(7, 7), dia/2+1, (1, 1, 1), -1);
            tSum = sum(filter)[0];
            filter = filter/tSum;
            filter2D(subImg, subImg, -1, filter);
            //subImg.copyTo(bImg(Rect(i, j, size, size)));
            subImg.copyTo(bImg.colRange(j, j+size).rowRange(i, i+size));
            //bImg(Range(i, i+size), Range(j, j+size)) = subImg;
        }
    }
    //bImg.copyTo(foreground);
    //imshow("bokeh", bImg);
    imwrite("bokehImageRelative.png", bImg);
    bitwise_and(bImg, foreground, foreground);
    return 1;
}

int doMultiBlur(Mat img, Mat& retVal, Mat disp, Point p1)
{
    /* Implement multiple Gaussian blurs based on depth range*/
    int dispval, range, i, lval, hval;
    int l1, l2, h1, h2;
    vector<Mat> layers, blurs, finLayers;
    //Mat thresh, blur, bitwiseImg;
    int threshVal;



    LOGD("a");
    char str[10];
    char str2[]={"blur"};
    dispval = disp.at<uchar>(p1.y, p1.x);
    range = dispval/10;
    //printf("%d %d\n", range, dispval);
    lval = dispval+1;
    hval = dispval-1;
    for(i=1; i<4; i++)
    {
        l1 = lval - range;
        l2 = lval;
        h1 = hval;
        h2 = hval + range;
        //printf("%d %d %d %d\n", l1, l2, h1, h2);
        Mat thresh;
        Mat seg;

        /* Get thresholded layers */
        threshVal = getThresh(disp, thresh, l1, l2, h1, h2);
        if (!threshVal)
        {
            //printf("break\n");
            break;
        }
        thresh.copyTo(seg);

        /* segment thresholded layers to get foreground */
        segmentBlurs(thresh, seg);
        //imshow("thresh", thresh);
        //imshow("seg", seg);
        //waitKey(0);
        layers.push_back(seg);
        imwrite("/mnt/sdcard/Studio3D/img_jni_seg.png", seg);


        lval = l1;
        hval = h2;
        range*=2;
    }
    LOGD("b");
    /* first layer is not to be blurred */
    blurs.push_back(img);
    imwrite("/mnt/sdcard/Studio3D/img_jni_img.png", img);

    width=img.rows; // don't even know why

    /* blur layers */
    for(i=1; i<layers.size(); i++)
    {
        Mat blur;
        GaussianBlur(img, blur, Size(19, 19), 2*i);
        //doCircBlur(img, blur, 3*i);
        //imshow("blur", blur);
        //waitKey(0);
        blurs.push_back(blur);
    }
    int sigma = 2*i;
    Mat backLayer;
    LOGD("c");

    backLayer = Mat::zeros(img.rows, img.cols, CV_8UC3);
    /* get masks of each blur layers */
    for(i=1; i<layers.size(); i++)
    {
        Mat bitwiseImg;
        bitwise_and(layers[i], blurs[i], bitwiseImg);
        LOGD("d");
        add(backLayer, layers[i], backLayer);
        //imshow("thresh", layers[i]);

        /* add mask layers */
        finLayers.push_back(bitwiseImg);
    }
    LOGD("d");
    //imshow("backLayer", backLayer);
    //waitKey(0);

    /* Blur whatever is remaining */
    Mat blurImage;
    backLayer = Scalar(255, 255, 255) - backLayer;
    LOGD("backLayer");
    GaussianBlur(img, blurImage, Size(19, 19), sigma);
    LOGD("GaussianBlur");
    bitwise_and(blurImage, backLayer, backLayer);
    LOGD("bitwise_and");

    /* add final layer */
    finLayers.push_back(backLayer);
    LOGD("stackUp begin");

    /* stack all the layers together to get final image */
    stackUp(finLayers, retVal);
    LOGD("stackUp done");

    LOGD("e");
    return 1;

}

int getThresh(Mat img, Mat& retVal, int l1, int l2, int h1, int h2)
{
    /* get threshold image based on range */
    Mat thresh1, thresh2, thresh;
    if (l2 < 0 && h1 > 255)
    {
        return 0;
    }

    if (l1 < 0 && l2 < 0)
    {
        thresh1 = Mat::zeros(img.size(), CV_8U);
    }
    else if(l1 < 0)
    {
        l1 = 0;
        inRange(img, l1, l2-1, thresh1);
    }
    else
    {
        inRange(img, l1, l2-1, thresh1);
    }

    if (h2 > 255 && h1 > 255)
    {
        thresh2 = Mat::zeros(img.size(), CV_8U);
    }
    else if (h2 > 255)
    {
        h2 = 255;
        inRange(img, h1+1, h2, thresh2);
    }
    else
    {
        inRange(img, h1+1, h2, thresh2);
    }

    bitwise_or(thresh1, thresh2, thresh);

    vector<Mat> threshLayers;
    threshLayers.push_back(thresh);
    threshLayers.push_back(thresh);
    threshLayers.push_back(thresh);

    merge(threshLayers, retVal);

    if (retVal.size() == img.size())
    {
        return 1;
    }
    return 0;
}

int getGaussianBlur(Mat img, Mat& retVal, int ksize)
{
    /* gaussian blur */
    GaussianBlur(img, retVal, Size(ksize, ksize), 0);
    if (retVal.size() == img.size())
    {
        return 1;
    }
    return 0;
}

int stackUp(vector<Mat>& layers, Mat& retVal)
{

    /* add all the layers to form final image */

  LOGD("in stackup");
    int i;
    //Mat zerotemp;
    //zerotemp = Mat::zeros(layers[i].size(), CV_8UC3);
    //zerotemp.copyTo(retVal);
    char str[10];
    char str3[] = {"stackUp"};

    retVal = Mat::zeros(layers[i].size(), CV_8UC3);
    for(i=0; i<layers.size(); i++)
    {
        sprintf(str, "%d", layers[i].channels());
        strcat(str, str3);
        LOGD(str);
        sprintf(str, "%d", retVal.channels());
        strcat(str, str3);
        LOGD(str);
        sprintf(str, "%d", layers[i].rows);
        strcat(str, str3);
        LOGD(str);
        sprintf(str, "%d", retVal.rows);
        strcat(str, str3);
        LOGD(str);
        LOGD("adding layer");
        add(retVal, layers[i], retVal);
    }
    if (retVal.size() == layers[0].size())
    {
        return 1;
    }
    return 0;
}

int doCircBlur(Mat img, Mat& retVal, int radius)
{
    Mat circ;
    int tSum;

    circ = Mat::zeros(31, 31, CV_64F);
    circle(circ, Point(15, 15), radius, (1, 1, 1), -1);
    tSum = sum(circ)[0];
    circ = circ/tSum;

    filter2D(img, retVal, -1, circ);
    if (retVal.size() == img.size())
    {
        return 1;
    }
    return 0;
}

int getDisparity(Mat g1, Mat g2, Mat &disp)
{
    /* not used */
    Mat disp16;
    StereoSGBM sbm;
    sbm.SADWindowSize = 7; // 5
    sbm.numberOfDisparities = 128;
    sbm.preFilterCap = 4;
    sbm.minDisparity = -39; // -64
    sbm.uniquenessRatio = 9; // 1
    sbm.speckleWindowSize = 180; //150
    sbm.speckleRange = 2;
    sbm.disp12MaxDiff = 20; // 10
    sbm.fullDP = false;
    sbm.P1 = 600;
    sbm.P2 = 2400;
    Mat preBilateralDisp;
    sbm(g1, g2, disp16);
    normalize(disp16, preBilateralDisp, 0, 255, CV_MINMAX, CV_8U);
    bilateralFilter(preBilateralDisp, disp, 5, 50, 0);
    if (disp.cols > 0 && disp.rows > 0)
    {
        return 1;
    }
    return 0;
}

int segmentBlurs(Mat img, Mat &foreground)
{
    /* segment blurs but without erode and dilate */
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat drawing, kernel;
    int size=3;
    drawing = Mat::zeros(img.size(), CV_8UC3);
    Mat fg;
    cvtColor(foreground, fg, CV_BGR2GRAY);
    findContours(fg.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 3000)
        {
            //printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    /*
    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));
    //erode(drawing, drawing, kernel, Point(-1, -1), 2);
    Mat temp;
    //dilate(drawing, temp, kernel, Point(-1, -1), 1);
    temp = drawing.clone();
    drawing = Mat::zeros(img.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    cvtColor(temp, temp, CV_BGR2GRAY);
    findContours(temp.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 2000)
        {
            printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }*/
    //dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    foreground = drawing.clone();
    //imshow("contours", drawing);
    //background = Scalar(255, 255, 255) - foreground;

    drawing = Mat::zeros(img.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    return 1;
}

int doOilPaint(Mat src, Mat& foreground)
{
    /* Oil Paint effect */
    Mat dst;
    int nRadius = 5;
    int fIntensityLevels = 20;

    dst = Mat::zeros(src.size(), src.type());

    for(int nY = nRadius; nY < (src.rows - nRadius); nY++)
    {
        for(int nX = nRadius; nX < (src.cols - nRadius); nX++)
        {
            int nIntensityCount[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            int nSumB[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            int nSumG[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            int nSumR[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            for(int nY_O = -nRadius; nY_O <= nRadius; nY_O++)
            {
                for(int nX_O = -nRadius; nX_O <= nRadius; nX_O++)
                {
                    int nB = src.at<Vec3b>((nY+nY_O), (nX+nX_O))[0];
                    int nG = src.at<Vec3b>((nY+nY_O), (nX+nX_O))[1];
                    int nR = src.at<Vec3b>((nY+nY_O), (nX+nX_O))[2];

                    int nCurIntensity = (((nB+nG+nR)/3.0)*fIntensityLevels)/255;
                    if(nCurIntensity > 255)
                    {
                        nCurIntensity = 255;
                    }
                    int i = nCurIntensity;
                    nIntensityCount[i]++;

                    nSumB[i] = nSumB[i] + nB;
                    nSumG[i] = nSumG[i] + nG;
                    nSumR[i] = nSumR[i] + nR;

                }
            }

            int nCurMax = 0;
            int nMaxIndex = 0;

            for(int nI = 0; nI<21; nI++)
            {
                if(nIntensityCount[nI] > nCurMax)
                {
                    nCurMax = nIntensityCount[nI];
                    nMaxIndex = nI;
                }
            }

            dst.at<Vec3b>(nY, nX)[0] = nSumB[nMaxIndex]/nCurMax;
            dst.at<Vec3b>(nY, nX)[1] = nSumG[nMaxIndex]/nCurMax;
            dst.at<Vec3b>(nY, nX)[2] = nSumR[nMaxIndex]/nCurMax;
        }
    }
    bitwise_and(dst, foreground, foreground);
    return 1;
}

int doGraySingle(Mat img, Mat& retVal, Mat disp, Point p1)
{
    /* no idea what this does. Don't remember. Not used */
    int dispval, range;
    int l1, l2, h1, h2;
    dispval = disp.at<uchar>(p1.y, p1.x);
    range = getRange(disp, p1);

    Mat thresh;
    l1 = dispval - range;
    l2 = dispval;
    h1 = dispval;
    h2 = dispval + range;

    inRange(disp, dispval - range, dispval + range, thresh);
    vector<Mat> threshvec;
    threshvec.push_back(thresh);
    threshvec.push_back(thresh);
    threshvec.push_back(thresh);
    merge(threshvec, thresh);

    bitwise_and(img, thresh, retVal);

    thresh = Scalar(255, 255, 255) - thresh;

    Mat gray;
    cvtColor(img, gray, CV_BGR2GRAY);

    vector<Mat> grayvec;
    grayvec.push_back(gray);
    grayvec.push_back(gray);
    grayvec.push_back(gray);
    merge(grayvec, gray);
    bitwise_and(gray, thresh, gray);
    add(gray, retVal, retVal);
  return 1;
}

int getRange(Mat disp, Point p1)
{
    /* some "advanced" way of getting thresholding range. Don't if used 
     * anymore or not */
    int dispval, range;
    dispval = disp.at<uchar>(p1.y, p1.x);
    range = dispval/8;
    Mat thresh;

    inRange(disp, dispval - range, dispval + range, thresh);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat drawing, kernel;
    int size=3;
    drawing = Mat::zeros(thresh.size(), CV_8UC3);
    Mat fg;
    //cvtColor(foreground, fg, CV_BGR2GRAY);
    findContours(thresh.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 1000)
        {
            //printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }

    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));
    //erode(drawing, drawing, kernel, Point(-1, -1), 2);
    Mat temp;
    //dilate(drawing, temp, kernel, Point(-1, -1), 1);
    temp = drawing.clone();
    drawing = Mat::zeros(thresh.size(), CV_8UC1);
    contours.clear();
    hierarchy.clear();
    cvtColor(temp, temp, CV_BGR2GRAY);
    findContours(temp.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 2000)
        {
            //printf("%lf\n", contourArea(contours[i]));
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    //dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    //foreground = drawing.clone();
    //imshow("contours", drawing);
    //background = Scalar(255, 255, 255) - foreground;

    Mat dispClone;
    bitwise_and(disp, drawing, dispClone);
    //imshow("dispClone", dispClone);
    Point minP, maxP;
    double minVal, maxVal;

    minMaxLoc(dispClone, &minVal, &maxVal, &minP, &maxP, thresh);
    //printf("%lf %lf\n", minVal, maxVal);

    //drawing = Mat::zeros(thresh.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    return (maxVal - dispval);
}

int stickImage(Mat &foreground, Mat &background)
{
    /* stick some different foreground in the image */
    Mat bitwise;
    int i, j;
    resize(background, background, Size(foreground.cols, foreground.rows));
    for(i=0; i<background.rows; i++)
    {
        for(j=0; j<background.cols; j++)
        {
            if (foreground.at<Vec3b>(i,j)[0] != 0)
            {
                background.at<Vec3b>(i,j) = foreground.at<Vec3b>(i,j);
            }
        }
    }

}

int histPick(Mat disp)
{
    /* function to get disparity value based on histogram of disparity.
     * Better function is present. No need to use this */
    int pxv[256] = {0};

//    char str[10];
//     char str2[]={" Value"};
//     sprintf(str, "%d", tLen);
//     strcat(str,str2);

    for(int i=0; i<disp.cols; i++)
    {
        for(int j=0; j<disp.rows; j++)
        {
            pxv[disp.at<uchar>(j, i)] += 1;

//            LOGD("rows ");
        }
    }
//LOGD("float initialization");
    int maxval=0;
    int val;
    int maxindex = 255;

    for(int i=230; i>100; i--)
    {
        val = pxv[i];
        if (val > maxval)
        {
            maxval = val;
            maxindex = i;
        }
    }

    LOGD("max index found");
    return (maxindex);
}

int doMultiBlurHist(Mat img, Mat& retVal, Mat disp, int dispval)
{
    /* Multiple Gaussian Blur based on histogram */

    int range, i, lval, hval;
    int l1, l2, h1, h2;
    vector<Mat> layers, blurs, finLayers;
    //Mat thresh, blur, bitwiseImg;
    int threshVal;

    range = dispval/10;
    //printf("%d %d\n", range, dispval);
    lval = dispval+1;
    hval = dispval-1;
    for(i=1; i<4; i++)
    {
        l1 = lval - range;
        l2 = lval;
        h1 = hval;
        h2 = hval + range;

        //printf("%d %d %d %d\n", l1, l2, h1, h2);

        Mat thresh;
        Mat seg;
        threshVal = getThresh(disp, thresh, l1, l2, h1, h2);

        if (!threshVal)
        {
            //printf("break\n");
            break;
        }
        thresh.copyTo(seg);
        segmentBlurs(thresh, seg);
        //imshow("thresh", thresh);
        //imshow("seg", seg);
        //waitKey(0);
        layers.push_back(seg);

        lval = l1;
        hval = h2;
        range*=2;
    }

    blurs.push_back(img);
    for(i=1; i<layers.size(); i++)
    {
        Mat blur;
        GaussianBlur(img, blur, Size(19, 19), 2*i);
        //doCircBlur(img, blur, 3*i);
        //imshow("blur", blur);
        //waitKey(0);
        blurs.push_back(blur);
    }
    int sigma = 2*i;
    Mat backLayer;
    backLayer = Mat::zeros(img.rows, img.cols, CV_8UC3);
    for(i=1; i<layers.size(); i++)
    {
        Mat bitwiseImg;
        //printf("%d %d %d %d\n", layers[i].cols, layers[i].rows, blurs[i].rows, blurs[i].cols);
        //printf("%d %d\n", layers[i].channels(), blurs[i].channels());
        bitwise_and(layers[i], blurs[i], bitwiseImg);
        //imshow("bitwiseImg", bitwiseImg);
        //waitKey(0);
        //printf("%d %d %d %d\n", layers[i].cols, layers[i].rows, backLayer.rows, backLayer.cols);
        add(backLayer, layers[i], backLayer);
        //imshow("thresh", layers[i]);
        
        finLayers.push_back(bitwiseImg);
    }
    //imshow("backLayer", backLayer);
    //waitKey(0);
    Mat blurImage;
    backLayer = Scalar(255, 255, 255) - backLayer;
    GaussianBlur(img, blurImage, Size(19, 19), sigma);
    bitwise_and(blurImage, backLayer, backLayer);
    finLayers.push_back(backLayer);
    stackUp(finLayers, retVal);

    return 1;

}

int getThresholdHist(Mat img, int dispval, int range, Mat &foreground)
{
    range = dispval/10;
    inRange(img, dispval - range, dispval + range, foreground);
    medianBlur(foreground, foreground, 9);
    return 1;
}

int histPick(Mat Oirgdisp, int &lower, int &upper)
{
    /* Better way to get disparity value and range using histogram
     * of disparit map.
     * It looks for peaks in the predefined range (needs to be changed)
     * It finds range based on number of pixels in the neighborhood
     */

    // TODO: Find peaks and set range

    Mat disp(Oirgdisp, Rect(150, 0, Oirgdisp.cols - 300, Oirgdisp.rows));
    //imshow("cropped", disp);
    int pxv[256] = {0};
    //printf("%d %d\n", disp.rows, disp.cols);
    for(int i=0; i<disp.rows; i++)
    {
        for(int j=0; j<disp.cols; j++)
        {
            pxv[disp.at<uchar>(i, j)] += 1;
        }
    }
    //printf("histogram created\n");
    /*for(int i=0; i<256; i++)
    {
        printf("%d - %d\n", i, pxv[i]);
    }*/

    int maxval=0;
    int val;
    int maxindex = 255;
    int sumPix=0;
    int totalPix = disp.cols * disp.rows;

    for(int i=255; i>200; i--)
    {
        val = pxv[i];
        sumPix += val;
        if (val > maxval)
        {
            maxval = val;
            maxindex = i;
        }
    }
    //printf("interim dispval = %d\n", maxindex);
    //printf("pixVal = %d\n", pxv[maxindex]);
    //printf("sumPix = %d totalPix = %d\n", sumPix, totalPix);

    if (pxv[maxindex] < sumPix/20 || sumPix < totalPix/10)
    {
        //printf("not enough pixels to support this value. Increasing the range.\n");
        maxval=0;
        maxindex = 200;
        sumPix = 0;

        for(int i=200; i>100; i--)
        {
            val = pxv[i];
            sumPix += val;
            if (val > maxval)
            {
                maxval = val;
                maxindex = i;
            }
        }

        //printf("interim dispval = %d\n", maxindex);
        //printf("pixVal = %d\n", pxv[maxindex]);
        //printf("sumPix = %d totalPix = %d\n", sumPix, totalPix);

        if (pxv[maxindex] < sumPix/20 || sumPix < totalPix/20)
        {
            //printf("not enough pixels to support this value. Increasing the range.\n");
            maxval=0;
            maxindex = 100;
            sumPix = 0;

            for(int i=100; i>50; i--)
            {
                val = pxv[i];
                sumPix += val;
                if (val > maxval)
                {
                    maxval = val;
                    maxindex = i;
                }
            }

            //printf("interim dispval = %d\n", maxindex);
            //printf("pixVal = %d\n", pxv[maxindex]);
            //printf("sumPix = %d totalPix = %d\n", sumPix, totalPix);
        }

    }
    histRange(pxv, maxindex, lower, upper);
    return (maxindex);}

int histRange(int pxv[], int index, int& lower, int& upper)
{
    /* To find range based on histogram and disparity value.
     * from the disparity value, it checks in left and right 
     * and whenever the index has pixels less than pixels/10
     * at disparity value, it increments tCount. When tCount
     * is equal to tolerance, the loop breaks and the last saved
     * value is kept as lower or upper range
     */
    int pixCount = pxv[index];
    int threshval = pixCount/10;
    int tolerance=3;
    int tCount=0;

    upper = index;
    lower = index;
    for(int i=index; i<256; i++)
    {
        //printf("val = %d, threshval = %d\n", pxv[i], threshval);
        if (pxv[i] < threshval)
        {
            tCount++;
        }
        else
        {
            tCount = 0;
            upper = i;
        }

        if(tCount > tolerance)
        {
            break;
        }
    }
    tCount = 0;
    for(int i=index; i>0; i--)
    {
        //printf("val = %d, threshval = %d\n", pxv[i], threshval);
        if (pxv[i] < threshval)
        {
            tCount++;
        }
        else
        {
            tCount = 0;
            lower = i;
        }

        if(tCount > tolerance)
        {
            break;
        }
    }

    if (upper != lower && upper != index)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int histSegmentForeground(Mat &foreground, Mat &background)
{

    /* segment foregoround and background */
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat drawing, kernel;
    int size=3;
    drawing = Mat::zeros(foreground.size(), CV_8UC3);
    findContours(foreground.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 15000)
        {
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }

    kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*size+1, 2*size+1), Point(size, size));
    //erode(drawing, drawing, kernel, Point(-1, -1), 2);
    Mat temp;
    drawing.copyTo(temp);
    //dilate(drawing, temp, kernel, Point(-1, -1), 1);
    drawing = Mat::zeros(foreground.size(), CV_8UC3);
    contours.clear();
    hierarchy.clear();
    cvtColor(temp, temp, CV_BGR2GRAY);
    findContours(temp.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int i=0; i<contours.size(); i++)
    {
        if (contourArea(contours[i]) > 15000)
        {
            drawContours(drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());
        }
    }
    //dilate(drawing, drawing, kernel, Point(-1, -1), 1);
    foreground = drawing.clone();
    background = Scalar(255, 255, 255) - foreground;
    return 1;
}

