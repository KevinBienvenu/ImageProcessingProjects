#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <math.h>  

#include "maxflow/graph.h"

using namespace std;
using namespace cv;

// This section shows how to use the library to compute a minimum cut on the following graph:
//
//		        SOURCE
//		       /       \
//		     1/         \6
//		     /      4    \
//		   node0 -----> node1
//		     |   <-----   |
//		     |      3     |
//		     \            /
//		     5\          /1
//		       \        /
//		          SINK
//
///////////////////////////////////////////////////



void testGCuts()
{
	
	//Graph<int,int,int> g(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1);
	//g.add_node(2); 
	//g.add_tweights( 0,   /* capacities */  1, 5 );
	//g.add_tweights( 1,   /* capacities */  6, 1 );
	//g.add_edge( 0, 1,    /* capacities */  4, 3 );
	
	namedWindow("images");

	Mat I=imread("../fishes.jpg");
	imshow("I", I);

	int channels = I.channels();
	int m = I.rows;
	int n = I.cols;
	float grad;
	int c;
	Graph<int,int,int> g(n*m, 2*n*m);
	g.add_node(n*m);

	
	//On ajoute des graines d'entrée et de sortie
	int iEntree = 1;
	int jEntree = 1;
	int iSortie = m/4;
	int jSortie = n/4;

	//On crée les arrêtes de voisinage

	float maxgrad=0;
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			//Avec le voisin de droite
			if(j<n-1){
				grad = float(I.at<uchar>(i,j*channels))-float(I.at<uchar>(i,(j+1)*channels));
				if(grad<0) grad = grad*(-1);
				if(grad>maxgrad) maxgrad = grad;
			}
			//Avec le voisin du bas
			if(i<m-1){
				grad = float(I.at<uchar>(i,j*channels))-float(I.at<uchar>(i+1,j*channels));
				if(grad<0) grad = grad*(-1);
				if(grad>maxgrad) maxgrad = grad;
			}
			//Avec le voisin du bas-droit
			if(i<m-1 && j<n-1){
				grad = float(I.at<uchar>(i,j*channels))-float(I.at<uchar>(i+1,(j+1)*channels));
				if(grad<0) grad = grad*(-1);
				if(grad>maxgrad) maxgrad = grad;
			}
			//Avec le voisin du haut-droit
			if(i>0 && j<n-1){
				grad = float(I.at<uchar>(i,j*channels))-float(I.at<uchar>(i-1,(j+1)*channels));
				if(grad<0) grad = grad*(-1);
				if(grad>maxgrad) maxgrad = grad;
			}
		}
	}
	cout<<maxgrad<<endl;
	maxgrad= maxgrad;
	float gradEntree, gradSortie;
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			gradEntree = float(I.at<uchar>(i,j*channels))-float(I.at<uchar>(iEntree,jEntree));
			gradSortie = float(I.at<uchar>(i,j*channels))-float(I.at<uchar>(iSortie,jSortie));
			if(gradEntree<0) gradEntree = gradEntree*(-1);
			if(gradEntree==0) gradEntree = 1;
			if(gradSortie<0) gradSortie = gradSortie*(-1);
			if(gradSortie==0) gradSortie = 1;
			//g.add_tweights(i*n+j,maxgrad/(gradEntree*gradEntree),maxgrad/(gradSortie*gradSortie));
			
			maxgrad = maxgrad;
			
			//Avec le voisin de droite
			if(j<n-1){
				grad = float(I.at<uchar>(i,j*channels))-float(I.at<uchar>(i,(j+1)*channels));
				if(grad<0) grad = grad * (-1);
				if(grad==0) grad = 1;
				c = maxgrad/(grad*grad*3);
				//cout<<c << " " << grad << " " << maxgrad<<endl;
				g.add_edge(j+i*n, j+i*n+1,c,c);
			}
			//Avec le voisin du bas
			if(i<m-1){
				grad = float(I.at<uchar>(i,j*channels))-float(I.at<uchar>(i+1,j*channels));
				if(grad<0) grad = grad * (-1);
				if(grad==0) grad = 1;
				c = maxgrad/(grad*grad*3);
				//cout<<c << " " << grad << " " << maxgrad<<endl;
				g.add_edge(j+i*n, j+i*n+n,c,c);
			}
			//Avec le voisin du bas-droit
			if(i<m-1 && j<n-1){
				grad = float(I.at<uchar>(i,j*channels))-float(I.at<uchar>(i+1,(j+1)*channels));
				if(grad<0) grad = grad * (-1);
				if(grad==0) grad = 1;
				c = maxgrad/(grad*grad*3);
				//cout<<c << " " << grad << " " << maxgrad<<endl;
				g.add_edge(j+i*n, j+i*n+n+1,c,c);
			}
			//Avec le voisin du haut-droit
			if(i>0 && j<n-1){
				grad = float(I.at<uchar>(i,j*channels))-float(I.at<uchar>(i-1,(j+1)*channels));
				if(grad<0) grad = grad * (-1);
				if(grad==0) grad = 1;
				c = maxgrad/(grad*grad*3);
				//cout<<c << " " << grad << " " << maxgrad<<endl;
				g.add_edge(j+i*n, j+i*n-n+1,c,c);
			}
		}
	}

	g.add_tweights(iEntree*n + jEntree, 1, 8*maxgrad);
	g.add_tweights(iSortie*n + jSortie, 8*maxgrad, 1);

	int flow = g.maxflow();
	cout << "Flow = " << flow << endl;
	for (int i=0;i<m;i++){
		for(int j=0; j<n; j++){
			if (g.what_segment(i*n+j) != Graph<int,int,int>::SOURCE){
				I.at<uchar>(i,j*channels)=0;
				I.at<uchar>(i,j*channels+1)=0;
				I.at<uchar>(i,j*channels+2)=0;
			}
			
		}
	}
	Mat boo(m,n,CV_32F);
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			if(I.at<uchar>(i,j*channels)==0) boo.at<float>(i,j)=0;
			else boo.at<float>(i,j)=1;
			if(i==iSortie && j==jSortie) boo.at<float>(i,j)=2;
		}
	}
	bool b = true;
	while(b){
		b = false;
		for(int i=1; i<m-1; i++){
			for(int j=1; j<n-1; j++){
				if(boo.at<float>(i,j)==1){
					if(boo.at<float>(i,j+1)==2){
						boo.at<float>(i,j)=2;
						b = true;
					}
					if(boo.at<float>(i,j-1)==2){
						boo.at<float>(i,j)=2;
						b = true;
					}
					if(boo.at<float>(i+1,j)==2){
						boo.at<float>(i,j)=2;
						b = true;
					}
					if(boo.at<float>(i-1,j)==2){
						boo.at<float>(i,j)=2;
						b = true;
					}
				}
			}
		}
	}
	for (int i=0;i<m;i++){
		for(int j=0; j<n; j++){
			if (boo.at<float>(i,j) == 1){
				I.at<uchar>(i,j*channels)=0;
				I.at<uchar>(i,j*channels+1)=0;
				I.at<uchar>(i,j*channels+2)=0;
			}
			
		}
	}

	imshow("image modifiée",I);
}

int main() {
	testGCuts();

	waitKey(0);
	return 0;
}
