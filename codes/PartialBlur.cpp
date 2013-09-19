#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
	if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }
	
	Mat src; Mat dst;
	Mat src_mask; Mat dst_mask;
	Mat final;
	
	src = imread(argv[1], 1);
	
	if(! src.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
	
	dst = src.clone();
	final = src.clone();
	
	dst = Mat::zeros( src.size(), src.type() );
	final = Mat::zeros( src.size(), src.type() );
	
	namedWindow( "SRC", CV_WINDOW_AUTOSIZE );
	imshow( "SRC", src );
	
	GaussianBlur( src, dst, Size( 15, 15 ), 0, 0 );
	
	namedWindow( "DST", CV_WINDOW_AUTOSIZE );
	imshow( "DST", dst );
	
	src_mask = src(Rect(200, 200, 200, 200));
	
	dst_mask = Mat::zeros(src_mask.size(), src_mask.type());
	dst_mask.copyTo(dst(Rect(200, 200, 200, 200)));
	imshow("DST_MASK", dst_mask);
	dst.copyTo(final);
	src_mask.copyTo(final(Rect(200, 200, 200, 200)));
	imshow("SRC_MASK", src_mask);
	namedWindow( "FINAL", CV_WINDOW_AUTOSIZE );
	imshow( "FINAL", final );
	
	waitKey(0);
	return(0);
}
	
	
	
	
	
	
