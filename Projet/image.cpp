#include "image.h"
#include <iostream>

// Correlation
double mean(const Image<float>& I,Point m,int n) {
	double s=0;
	for (int j=-n;j<=n;j++)
		for (int i=-n;i<=n;i++) 
			s+=I(m+Point(i,j));
	return s/(2*n+1)/(2*n+1);
}

double corr(const Image<float>& I1,Point m1,const Image<float>& I2,Point m2,int n) {
	double M1=mean(I1,m1,n);
	double M2=mean(I2,m2,n);
	double rho=0;
	for (int j=-n;j<=n;j++)
		for (int i=-n;i<=n;i++) {
			rho+=(I1(m1+Point(i,j))-M1)*(I2(m2+Point(i,j))-M2);
		}
		return rho;
}

double NCC(const Image<float>& I1,Point m1,const Image<float>& I2,Point m2,int n) {
	if (m1.x<n || m1.x>=I1.width()-n || m1.y<n || m1.y>=I1.height()-n) return -1;
	if (m2.x<n || m2.x>=I2.width()-n || m2.y<n || m2.y>=I2.height()-n) return -1;
	double c1=corr(I1,m1,I1,m1,n);
	if (c1==0) return -1;
	double c2=corr(I2,m2,I2,m2,n);
	if (c2==0) return -1;
	return corr(I1,m1,I2,m2,n)/sqrt(c1*c2);
}


// Correlation with pre-computed means
Image<float> meanImage(const Image<float>& I,int n) {
	Image<float> meanI(I.width(),I.height(),CV_32F);
	
	double vartrav;
	for(int i=n; i< I.height()-n; i++){
		for(int j=n; j<I.width()-n; j++){
			vartrav=0;
			for (int jbis=-n;jbis<n+1;jbis++){
				for (int ibis=-n;ibis<n+1;ibis++){ 
					vartrav+=I.at<float>(i+ibis,j+jbis);
				}
			}
			meanI.at<float>(i,j) =(float)(vartrav/((2*n+1)*(2*n+1)));
		}
	}
	return meanI;
}



// Fonction intermédiaire
float corr(const Image<float>& I1,const Image<float>& meanI1,Point m1,const Image<float>& I2,const Image<float>& meanI2,Point m2,int n) {
	float rho=0, v1, v2;
	for (int j=-n;j<=n;j++)
		for (int i=-n;i<=n;i++) {
			//rho+=(I1(m1+Point(i,j))-meanI1(m1))*(I2(m2+Point(i,j))-meanI2(m2));
			v1 = I1(m1+Point(i,j))-meanI1(m1);
			v2 = I2(m2+Point(i,j))-meanI2(m2);
			rho += v2*v1;
		}
		return rho;
}

// Renvoit la matrice avec toutes les auto-correlations
Image<float> corrImage(const Image<float>& I, const Image<float>& meanI, int n){
	Image<float> corrI(I.width(), I.height(),CV_32F);
	for(int i=n+1; i<I.height()-n-1; i++){
		for(int j=n+1; j<I.width()-n-1; j++){
			//cout<< i << " " << j << endl;
			corrI.at<float>(i,j) = (float)(corr(I, meanI, Point(j,i), I, meanI, Point(j,i),n));
		}
	}
	return corrI;
}

double NCC(const Image<float>& I1,const Image<float>& meanI1,const Image<float>& corrI1,Point m1,const Image<float>& I2,const Image<float>& meanI2,const Image<float>& corrI2,Point m2,int n) {
	if (m1.x<n || m1.x>=I1.width()-n || m1.y<n || m1.y>=I1.height()-n) return -1;
	if (m2.x<n || m2.x>=I2.width()-n || m2.y<n || m2.y>=I2.height()-n) return -1;
	if (corrI1(m1)==0) return -1;
	if (corrI2(m2)==0) return -1;
	return corr(I1,m1,I2,m2,n)/sqrt(sqrt(corrI1(m1)*corrI1(m1))*sqrt(corrI2(m2)*corrI2(m2)));
}

