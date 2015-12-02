#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <queue>

using namespace cv;
using namespace std;


// From [minVal;maxVal] to [0;255]
Mat float2byte(const Mat& If) {
	double minVal, maxVal;
	minMaxLoc(If,&minVal,&maxVal);
	Mat Ib;
	If.convertTo(Ib, CV_8U, 255.0/(maxVal - minVal), -255.*minVal/(maxVal-minVal));
	return Ib;
} 

int main()
{
	// Basic queue functions
	Point p(12,13);
	Point q(14,13);
	queue<Point> Q;
	Q.push(p);
	Q.push(q);
	Q.push(Point(16,17));
	while (!Q.empty()) {
		cout << Q.front() << endl;
		Q.pop();
	}

	// Read and display
    Mat A=imread("../fruits.jpg");
	namedWindow("images");

	Vec3b c=A.at<Vec3b>(12,12);
	cout << c << endl;

	A.at<Vec3b>(12,12)=Vec3b(255,0,0); // Blue!
	imshow("images",A);	waitKey();

	// Gray
	Mat I;
	cvtColor(A,I,CV_BGR2GRAY);
	cout << int(I.at<uchar>(12,12)) << endl; // Blue => Gray=29 not 255/3
	imshow("images",I);waitKey();

	// Gradient
	int m=I.rows,n=I.cols;
	Mat Ix(m,n,CV_32F),Iy(m,n,CV_32F),G(m,n,CV_32F);
	for (int i=0;i<m;i++) {
		for (int j=0;j<n;j++){
			float ix,iy;
			if (i==0 || i==m-1)
				iy=0;
			else
				iy=(float(I.at<uchar>(i+1,j))-float(I.at<uchar>(i-1,j)))/2;
			if (j==0 || j==n-1)
				ix=0;
			else
				ix=(float(I.at<uchar>(i,j+1))-float(I.at<uchar>(i,j-1)))/2;
			Ix.at<float>(i,j)=ix;
			Iy.at<float>(i,j)=iy;
			G.at<float>(i,j)=sqrt(ix*ix+iy*iy);
		}
	}
	imshow("images",float2byte(Ix));waitKey();
	imshow("images",float2byte(Iy));waitKey();
	imshow("images",float2byte(G));waitKey();

	// Threshold
	Mat C;
	threshold(G,C,10,1,THRESH_BINARY);
	imshow("images",float2byte(C));waitKey();
	
	// 1) A faire sur l'image C, en utilisant G,Ix,Iy:
	// 	a) suppression des non maxima dans la direction du gradient
	// 	b) seuillage par hysteresis
	// 2)Question: par rapport au véritable Canny, qu'avons nous oublié?
	// ...............
	
	
	
	return 0;
}
