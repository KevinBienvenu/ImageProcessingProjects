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

	// Choice of the gradient min
	float k = 12;
	// Choice of the gradient max (for threshold)
	float kmax = 13;

	// Read and display
    Mat A=imread("../portrait.png");
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
			if (i==0 || i==m-1 || j==0 || j==n-1)
				iy=0;
			else
				iy=(float(I.at<uchar>(i+1,j))-float(I.at<uchar>(i-1,j)))/2;
			if (j==0 || j==n-1 || i==0 || i==m-1)
				ix=0;
			else
				ix=(float(I.at<uchar>(i,j+1))-float(I.at<uchar>(i,j-1)))/2;
			Ix.at<float>(i,j)=ix;		
			Iy.at<float>(i,j)=iy;
			G.at<float>(i,j)=sqrt(ix*ix+iy*iy);
			if(G.at<float>(i,j) < k)
				G.at<float>(i,j) = 0;
		}
	}
	
	imshow("images",float2byte(Ix));waitKey();
	imshow("images",float2byte(Iy));waitKey();
	imshow("images",float2byte(G));waitKey();

	// Suppression des non-maxima
	Mat C = G;
	for (int i=0;i<m;i++) {
		for (int j=0;j<n;j++){
			float g = C.at<float>(i,j);
			if(g>0){
				float ix = Ix.at<float>(i,j);
				float iy = Iy.at<float>(i,j);
				float absx = ix, absy = iy;
				int dir=0;
				//détermination de la direction du gradient
				if(ix<0) absx = -ix;
				if(iy<0) absy = -iy;
				if(absx>absy/2 && iy*ix>0 && ix>0) dir = 3;
				if(absx>absy/2 && iy*ix>0 && ix<0) dir = 7;
				if(absx>absy/2 && iy*ix<0 && ix>0) dir = 9;
				if(absx>absy/2 && iy*ix<0 && ix<0) dir = 1;
				if(absx<absy/2 && iy>0) dir = 2;
				if(absx<absy/2 && iy<0) dir = 8;
				if(absy<absx/2 && ix>0) dir = 6;
				if(absy<absx/2 && ix>0) dir = 4;
				//suppression des non-maxima
				switch(dir)
				{
				case 1 : 
					C.at<float>(i+1,j-1)=0;
					break;
				case 2 : 
					C.at<float>(i,j-1)=0;
					break;
				case 3 : 
					C.at<float>(i-1,j-1)=0;
					break;
				case 4 : 
					C.at<float>(i+1,j)=0;
					break;
				case 6 : 
					C.at<float>(i-1,j)=0;
					break;
				case 7 : 
					C.at<float>(i+1,j+1)=0;
					break;
				case 8 : 
					C.at<float>(i,j+1)=0;
					break;
				case 9 : 
					C.at<float>(i-1,j+1)=0;
					break;
				}
			}	
		}
	}

	//Seuillage par hysteresis

	
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			float g = C.at<float>(i,j);
			if(g>kmax)
				C.at<float>(i,j)=kmax;
			if(i==0 || j==0 || i==m || j==n)
				C.at<float>(i,j)=0;
		}
	}
	for(int i=1;i<m-1; i++){
		for(int j=1; j<n-1; j++){
			if(C.at<float>(i,j)>k && C.at<float>(i,j)<kmax){
				bool a = C.at<float>(i+1,j)==kmax;
				a = a || C.at<float>(i-1,j)==kmax;
				a = a || C.at<float>(i+1,j+1)==kmax;
				a = a || C.at<float>(i,j+1)==kmax;
				a = a || C.at<float>(i-1,j+1)==kmax;
				a = a || C.at<float>(i,j-1)==kmax;
				a = a || C.at<float>(i-1,j-1)==kmax;
				a = a || C.at<float>(i+1,j-1)==kmax;
				if(a)
					C.at<float>(i,j)=kmax;
				//décommenter pour ne plus afficher les contours faibles non sélectionnés.
				else
					C.at<float>(i,j)=0;
 			}
		}
	}


	//threshold(G,C,10,1,THRESH_BINARY);
	imshow("images",float2byte(C));waitKey();
	
	// 1) A faire sur l'image C, en utilisant G,Ix,Iy:
	// 	a) suppression des non maxima dans la direction du gradient
	// 	b) seuillage par hysteresis
	// 2)Question: par rapport au véritable Canny, qu'avons nous oublié?
	// ...............
	
	
	
	return 0;
}
