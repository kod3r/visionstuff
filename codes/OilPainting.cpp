#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat src, dst;
	int nRadius = 2;
	int fIntensityLevels = 20;
	
	src = imread( argv[1], 1 );
	if( !src.data )
    {
		cout<<"Usage: ./OilPainting <path_to_image>"<<endl;
		return -1;
	}
	
	//dst = src.clone();
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
	
	namedWindow( "Source", CV_WINDOW_AUTOSIZE );
	namedWindow( "Result", CV_WINDOW_AUTOSIZE );
	imshow( "Source", src );
	imshow( "Result", dst );
	
	waitKey(0);
	return(0);
}
