    #include <cv.h>
    #include <highgui.h>
    #include <math.h>
    #include <unistd.h>
    #include <getopt.h>
    #include <iostream>


    void sampleImage(const IplImage* arr, float idx0, float idx1, CvScalar& res)
    {
      if(idx0<0 || idx1<0 || idx0>(cvGetSize(arr).height-1) || idx1>(cvGetSize(arr).width-1)){
        res.val[0]=0;
        res.val[1]=0;
        res.val[2]=0;
        res.val[3]=0;
        return;
      }
      float idx0_fl=floor(idx0);
      float idx0_cl=ceil(idx0);
      float idx1_fl=floor(idx1);
      float idx1_cl=ceil(idx1);

      CvScalar s1=cvGet2D(arr,(int)idx0_fl,(int)idx1_fl);
      CvScalar s2=cvGet2D(arr,(int)idx0_fl,(int)idx1_cl);
      CvScalar s3=cvGet2D(arr,(int)idx0_cl,(int)idx1_cl);
      CvScalar s4=cvGet2D(arr,(int)idx0_cl,(int)idx1_fl);
      float x = idx0 - idx0_fl;
      float y = idx1 - idx1_fl;
      res.val[0]= s1.val[0]*(1-x)*(1-y) + s2.val[0]*(1-x)*y + s3.val[0]*x*y + s4.val[0]*x*(1-y);
      res.val[1]= s1.val[1]*(1-x)*(1-y) + s2.val[1]*(1-x)*y + s3.val[1]*x*y + s4.val[1]*x*(1-y);
      res.val[2]= s1.val[2]*(1-x)*(1-y) + s2.val[2]*(1-x)*y + s3.val[2]*x*y + s4.val[2]*x*(1-y);
      res.val[3]= s1.val[3]*(1-x)*(1-y) + s2.val[3]*(1-x)*y + s3.val[3]*x*y + s4.val[3]*x*(1-y);
    }

    float xscale;
    float yscale;
    float xshift;
    float yshift;

    float getRadialX(float x,float y,float cx,float cy,float k){
      x = (x*xscale+xshift);
      y = (y*yscale+yshift);
      float res = x+((x-cx)*k*((x-cx)*(x-cx)+(y-cy)*(y-cy)));
      return res;
    }

    float getRadialY(float x,float y,float cx,float cy,float k){
      x = (x*xscale+xshift);
      y = (y*yscale+yshift);
      float res = y+((y-cy)*k*((x-cx)*(x-cx)+(y-cy)*(y-cy)));
      return res;
    }

    float thresh = 1;
    float calc_shift(float x1,float x2,float cx,float k){
      float x3 = x1+(x2-x1)*0.5;
      float res1 = x1+((x1-cx)*k*((x1-cx)*(x1-cx)));
      float res3 = x3+((x3-cx)*k*((x3-cx)*(x3-cx)));

      //  std::cerr<<"x1: "<<x1<<" - "<<res1<<" x3: "<<x3<<" - "<<res3<<std::endl;

      if(res1>-thresh and res1 < thresh)
        return x1;
      if(res3<0){
        return calc_shift(x3,x2,cx,k);
      }
      else{
        return calc_shift(x1,x3,cx,k);
      }
    }

    int main(int argc, char** argv)
    {
      IplImage* src = cvLoadImage( argv[1], 1 );
      IplImage* dst = cvCreateImage(cvGetSize(src),src->depth,src->nChannels);
      IplImage* dst2 = cvCreateImage(cvGetSize(src),src->depth,src->nChannels);
      float K=atof(argv[3]);
      float centerX=atoi(argv[4]);
      float centerY=atoi(argv[5]);
      int width = cvGetSize(src).width;
      int height = cvGetSize(src).height;

      xshift = calc_shift(0,centerX-1,centerX,K);
      float newcenterX = width-centerX;
      float xshift_2 = calc_shift(0,newcenterX-1,newcenterX,K);

      yshift = calc_shift(0,centerY-1,centerY,K);
      float newcenterY = height-centerY;
      float yshift_2 = calc_shift(0,newcenterY-1,newcenterY,K);
      //  scale = (centerX-xshift)/centerX;
      xscale = (width-xshift-xshift_2)/width;
      yscale = (height-yshift-yshift_2)/height;

      std::cerr<<xshift<<" "<<yshift<<" "<<xscale<<" "<<yscale<<std::endl;
      std::cerr<<cvGetSize(src).height<<std::endl;
      std::cerr<<cvGetSize(src).width<<std::endl;

      for(int j=0;j<cvGetSize(dst).height;j++){
        for(int i=0;i<cvGetSize(dst).width;i++){
          CvScalar s;
          float x = getRadialX((float)i,(float)j,centerX,centerY,K);
          float y = getRadialY((float)i,(float)j,centerX,centerY,K);
          sampleImage(src,y,x,s);
          cvSet2D(dst,j,i,s);

        }
      }
    #if 0
      cvNamedWindow( "Source1", 1 );
      cvShowImage( "Source1", dst);
      cvWaitKey(0);
    #endif

      cvSaveImage(argv[2],dst,0);

    #if 0
      for(int j=0;j<cvGetSize(src).height;j++){
        for(int i=0;i<cvGetSize(src).width;i++){
          CvScalar s;
          sampleImage(src,j+0.25,i+0.25,s);
          cvSet2D(dst,j,i,s);
        }
      }

      cvNamedWindow( "Source1", 1 );
      cvShowImage( "Source1", src);
      cvWaitKey(0);

    #endif  

}