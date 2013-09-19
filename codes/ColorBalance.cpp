#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/core/core.hpp" 
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void contrast_stretch(Mat src, Mat dst, int min, int max, int index);

int main( int argc, char **argv)
{
	Mat src, dst, src1;
	
	src = imread( argv[1], 1 );

	if( !src.data )
	{ 
		cout<<"Usage: ./Histogram_Demo <path_to_image>"<<endl;
		return -1;
	}
	
	src.copyTo(dst);
	src.copyTo(src1);
	
	int n = src.rows*src.cols;
	float s1 = 1.5, s2 = 1.5;
	
	vector<Mat> bgr_planes;
	split(src, bgr_planes);
	
	int hist_size = 256;
	
	float range[] = {0, 256};
	const float* hist_range = {range};
	
	Mat bhist, ghist, rhist;
	float cbhist[hist_size], cghist[hist_size], crhist[hist_size];
	
	int vmin1 = 0, vmin2 = 0, vmin3 = 0;
	int vmax1 = 255, vmax2 = 255, vmax3 = 255;
	
	calcHist(&bgr_planes[0], 1, 0, Mat(), bhist, 1, &hist_size, &hist_range, true, false);
	calcHist(&bgr_planes[1], 1, 0, Mat(), ghist, 1, &hist_size, &hist_range, true, false);
	calcHist(&bgr_planes[2], 1, 0, Mat(), rhist, 1, &hist_size, &hist_range, true, false);
	
	for(int i=0; i<hist_size; i++)
	{
		if( i ==0 )
		{
			cbhist[i]=bhist.at<float>(i);
			cghist[i]=ghist.at<float>(i);
			crhist[i]=rhist.at<float>(i);			
		}
		else
		{
			cbhist[i]=(cbhist[i-1] + bhist.at<float>(i));
			cghist[i]=(cghist[i-1] + ghist.at<float>(i));
			crhist[i]=(crhist[i-1] + rhist.at<float>(i));
		}		
	}
	
	while( vmin1<255 && (cbhist[vmin1] <= (float)(n*s1/100)))
	{
		vmin1 = vmin1 + 1;
	}	
	while( vmax1<255 && (cbhist[vmax1] <= (float)(n*(1 - s2/100))) )
	{
		vmax1 = vmax1 - 1;
	}	
	if(vmax1<255)
	{
		vmax1 = vmax1 + 1;
	}
	
	while( vmin2<255 && (cghist[vmin2] <= (float)(n*s1/100)))
	{
		vmin2 = vmin2 + 1;
	}	
	while( vmax2<255 && (cghist[vmax2] <= (float)(n*(1 - s2/100))) )
	{
		vmax2 = vmax2 - 1;
	}	
	if(vmax2<255)
	{
		vmax2 = vmax2 + 1;
	}
	
	while( vmin3<255 && (crhist[vmin3] <= (float)(n*s1/100)))
	{
		vmin3 = vmin3 + 1;
	}	
	while( vmax3<255 && (crhist[vmax3] <= (float)(n*(1 - s2/100))) )
	{
		vmax3 = vmax3 - 1;
	}	
	if(vmax3<255)
	{
		vmax3 = vmax3 + 1;
	}
	//Mat src1;
    //src.copyTo(src1);
	
	contrast_stretch(src1, dst, vmin1, vmax1, 0);
	contrast_stretch(src1, dst, vmin2, vmax2, 1);
	contrast_stretch(src1, dst, vmin3, vmax3, 2);
	
	bgr_planes;
	split(src1, bgr_planes);
	
	cv::normalize(bgr_planes[0], bgr_planes[0], 0, 255, NORM_MINMAX);
	cv::normalize(bgr_planes[1], bgr_planes[1], 0, 255, NORM_MINMAX);
	cv::normalize(bgr_planes[2], bgr_planes[2], 0, 255, NORM_MINMAX);
	cv::merge(bgr_planes, src1);
	
	namedWindow("Image", CV_WINDOW_AUTOSIZE);
	cv::imshow("Image", src);
	namedWindow("Image Process", CV_WINDOW_AUTOSIZE);
	cv::imshow("Image Process", src1);
	namedWindow("Final", CV_WINDOW_AUTOSIZE);
	cv::imshow("Final", dst);
	
	waitKey(0);
	return 0;
}

void contrast_stretch(Mat src, Mat dst, int min, int max, int index)
{
	int norm[256];
	
	if(max<=min)
	{
		cv::Mat_<cv::Vec3b>::const_iterator it = src.begin<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::const_iterator itend = src.end<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::iterator itout = dst.begin<cv::Vec3b>();
		
		for(; it!=itend ; ++it, ++itout)
		{
			cv::Vec3b color1 = *it;
			color1[index] = 255/2;
			*itout = color1;
		}		
	}
	
	else
	{
		int i = 0;
		
		for(i = 0; i<min; i++)
		{
			norm[i] = 0;
		}
		for(i = min; i<max; i++)
		{
			norm[i] = ((i-min)*255/((max-min)+0.5));
		}
		for(i=max; i<256; i++)
		{
			norm[i] = 255;
		}
		
		cv::Mat_<cv::Vec3b>::const_iterator it = src.begin<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::const_iterator itend = src.end<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::iterator itout = dst.begin<cv::Vec3b>();
		
		for(; it!=itend ; ++it, ++itout)
		{
			cv::Vec3b color = *it;
			cv::Vec3b color1 = *itout;
			color1[index] = norm[color[index]];
			*itout = color1;
		}		
	}
}