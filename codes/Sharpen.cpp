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
	
	Mat image; Mat sharp_image;	
	image = imread(argv[1], 1);
	
	if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
	
	namedWindow( "IMAGE", CV_WINDOW_AUTOSIZE );
	imshow( "IMAGE", image );
	
	sharp_image = image.clone();
	sharp_image = Mat::zeros( image.size(), image.type() );
	
	GaussianBlur( image, sharp_image, Size(5,5), 5 );
	addWeighted( image, 1.5, sharp_image, -0.5, 0, sharp_image);
    //GaussianBlur(sharp_image, sharp_image, Size(3, 3), 0);
	
	namedWindow( "SHARP", CV_WINDOW_AUTOSIZE );
	imshow( "SHARP", sharp_image );
	
	waitKey(0);
	return(0);
}