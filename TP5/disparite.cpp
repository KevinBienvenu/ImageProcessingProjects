#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <list>
#include "image.h"

// Sauve un maillage triangulaire dans un fichier ply.
// v: sommets (x,y,z), f: faces (indices des sommets), col: couleurs (par sommet) 
bool savePly(const string& name,const vector<Point3f>& v, const vector<Vec3i>& f,const vector<Vec3b>& col) {
	assert(v.size()==col.size());
	ofstream out(name.c_str());
	if (!out.is_open()) {
		cout << "Cannot save " << name << endl;
		return false;
	}
	out << "ply" << endl
		<< "format ascii 1.0" << endl
		<< "element vertex " << v.size() << endl
		<< "property float x" << endl
		<< "property float y" << endl
		<< "property float z" << endl
		<< "property uchar red" << endl                   
		<< "property uchar green" << endl
		<< "property uchar blue" << endl
		<< "element face " << f.size() << endl
		<< "property list uchar int vertex_index" << endl  
		<< "end_header" << endl;
	for (int i=0;i<v.size();i++)
		out << v[i].x << " " << v[i].y << " " << v[i].z << " " << int(col[i][2]) << " " << int(col[i][1]) << " " << int(col[i][0]) << " " << endl;
	for (int i=0;i<f.size();i++)
		out << "3 " << f[i][0] << " " << f[i][1] << " " << f[i][2] << endl;
	out.close();
	return true;
}

// Points de Harris
void harris(const Image<float>& I, Mat Har, double sh, int blockSize) {
	
	cornerHarris(I, Har, blockSize, 5, 1, 4);
	double vartrav;
	bool booltrav;
	double min, max, moy=0,cont=0;
	for(int i=0; i<Har.rows; i++){
		for(int j=0; j<Har.cols; j++){
			vartrav = Har.at<float>(i,j);
			if(i==0 && j==0) min = vartrav;
			else if(min>vartrav) min = vartrav;
			if(i==0 && j==0) max = vartrav;
			else if(max<vartrav) max = vartrav;
			moy+=vartrav;
			cont++;
			if(vartrav<sh || i<blockSize || j<blockSize || i>Har.rows-blockSize || j>Har.cols-blockSize){
				booltrav = false;
				Har.at<float>(i,j) = 0;
			} 
			else {
				booltrav = true;
				for(int ibis=-1; ibis<2; ibis++){
					for(int jbis=-1; jbis<2; jbis++){
						booltrav = booltrav && vartrav >= Har.at<float>(i+ibis, j+jbis);
					}
				}
				if(booltrav){
					for(int ibis=-1; ibis<2; ibis++){
					    for(int jbis=-1; jbis<2; jbis++){
							if(!(ibis==0 && jbis==0)) Har.at<float>(i+ibis, j+jbis)=0;
							else Har.at<float>(i+ibis, j+jbis)=1;
						}
					}
				}
			}

		}
	}
	cout<<"min: " << min << endl;
	cout<<"max: " << max << endl;
	cout<<"moy: " << moy/cont << endl;

}

// Vérification des graines
int verifGraine(Image<float> F1, Image<float> F2, Mat Corner, double sg, double rg, int i, int j, int blockSize){
	double corr;
	double phi=-1., phibis=-1.;
	double xcor=-1;
	for(int xbal=blockSize; xbal<F1.width()-blockSize; xbal++){
		corr = NCC(F1,Point(i,j),F2,Point(i,xbal),blockSize);
		if(corr>phi){
			phibis=phi;
			phi=corr;
			xcor = xbal;
		} else if(corr>phibis){
			phibis = corr;
		}
	}
	if(phi>sg && phi/phibis>rg)
		return (int)xcor;
	else
		return -1;
};
     //deuxième version du programme utilisant les algo améliorés
int verifGraine(Image<float> F1, Image<float> M1, Image<float> C1, Image<float> C2, Image<float> M2, Image<float> F2, Mat Corner, double sg, double rg, int i, int j, int blockSize){
	double corr;
	double phi=-1., phibis=-1.;
	double xcor=-1;
	for(int xbal=blockSize; xbal<F1.width()-blockSize; xbal++){
		corr = NCC(F1,M1,C1,Point(j,i),F2,M2,C2,Point(xbal,i),blockSize);
		if(corr>phi){
			phibis=phi;
			phi=corr;
			xcor = xbal;
		} else if(corr>phibis){
			phibis = corr;
		}
	}
	if(phi>sg && phi/phibis>rg)
		return (int)xcor;
	else
		return -1;
};

struct Data{
	Image<Vec3b> I1, I2;
	Image<float> F1, F2;
	double seuilHarris, seuilCroissance, seuilGraine, ratioGraine;
	Mat Corner, Disparite, IdentiteDisp;
};


int main() {
	// On initialise D
	Data D;
	D.I1 = imread("../face00R.tif");
	D.I2 = imread("../face01R.tif");
	assert(D.I1.height()==D.I2.height());
	imshow("I1",D.I1);
	imshow("I2",D.I2);


	// On définit les seuils et le blocksize
	D.seuilHarris = -10000;
	D.seuilCroissance = -0.4;
	D.seuilGraine = 0.2;
	D.ratioGraine = 1.0001;
	double blockSize = 5;


	// On convertit les images pour les rendre en float
	Image<uchar>G1,G2;
	cvtColor(D.I1,G1,CV_BGR2GRAY);
	cvtColor(D.I2,G2,CV_BGR2GRAY);
	G1.convertTo(D.F1,CV_32F);
	G2.convertTo(D.F2,CV_32F);
	D.Corner = Mat::zeros(D.I1.rows, D.I1.cols, CV_32F);
	D.Disparite = Mat::zeros(D.I1.rows, D.I1.cols, CV_32F);
	D.IdentiteDisp = Mat::zeros(D.I1.rows, D.I1.cols, CV_32F);

	cout << "fin des déclarations" << endl;
	

	// On calcule les matrices intermédiaires qui nous serviront à simplifier les calculs redondants de NCC
	Image<float> meanI1 = meanImage(D.F1,(int)blockSize);
	Image<float> meanI2 = meanImage(D.F2,(int)blockSize);
	Image<float> corrI1 = corrImage(D.F1,meanI1,(int)blockSize);
	Image<float> corrI2 = corrImage(D.F2,meanI2,(int)blockSize);

	cout << "fin des initialisations des matrices intermédiaires" << endl;

	// On appelle la fonction qui nous renvoit les points de Harris et on les affiche en rouge
	harris(D.F1,D.Corner,D.seuilHarris,5);
	cout << "Corner: " << D.Corner.rows << " " << D.Corner.cols << endl;
	cout << "F1: " << D.F1.rows << " " << D.F1.cols << endl;
	cout << "I1: " << D.I1.rows << " " << D.I1.cols << endl;
	cout << "I1: " << D.I1.height() << " " << D.I1.width() << endl;
	for(int i=(int)blockSize; i<(int)(D.Corner.rows-blockSize); i++){
		for(int j=(int)blockSize; j<(int)(D.Corner.cols-blockSize); j++){
			if(D.Corner.at<float>(i,j)==1){
				circle(D.I1,Point(j,i),2,Scalar(0,0,255),2);
			}
		}
	}

	// Désormais, on cherche lesquels pourront être des graines et on les affiche en vert
	int inttrav;
	for(int i=(int)blockSize; i<(int)(D.Corner.rows-blockSize); i++){
		for(int j=(int)blockSize; j<(int)(D.Corner.cols-blockSize); j++){
			if(D.Corner.at<float>(i,j)==1){
				//décommenter pour voir la première version non optimisée du programme
				//inttrav = verifGraine(D.F1,D.F2,D.Corner,D.seuilGraine,D.ratioGraine,i,j,(int)blockSize);
				inttrav = verifGraine(D.F1,meanI1,corrI1,corrI2,meanI2,D.F2,D.Corner,D.seuilGraine,D.ratioGraine,i,j,blockSize);
					(D.F1,D.F2,D.Corner,D.seuilGraine,D.ratioGraine,i,j,(int)blockSize);
				if(inttrav!=-1){
					D.Disparite.at<float>(i,j)=(float)(j-inttrav);		
					D.IdentiteDisp.at<float>(i,j) = 1;
				}
			}
		}
	}
	for(int i=(int)blockSize; i<(int)(D.Disparite.rows-blockSize); i++){
		for(int j=(int)blockSize; j<(int)(D.Disparite.cols-blockSize); j++){
			if(D.Disparite.at<float>(i,j)>0){
				circle(D.I1,Point(j,i),2,Scalar(0,255,0),2);
			}
		}
	}
	
	imshow("I1",D.I1);
	cout << "fin de l'affichage des graines" << endl;

	// Déterminons alors la disparité sur l'ensemble de l'image.
	// Pour cela nous utiliserons une matrice booleene IdentiteDisp des points dont on est sûr de la disparité
	// Ensuite nous compléterons au fur et à mesure la matrice Disparite

	bool switc = true;
	double corrminus, correqu, corrplus;

	while(switc){
		switc = false;
		for(int i=(int)blockSize+1; i<(int)(D.I1.rows-1-blockSize); i++){
			for(int j=(int)blockSize+1; j<(int)(D.I1.cols-1-blockSize); j++){
				if(D.IdentiteDisp.at<float>(i,j)==1){
					for(int ibis=-1; ibis<2; ibis++){
						for(int jbis=-1; jbis<2; jbis++){
							if(D.IdentiteDisp.at<float>(i+ibis,j+jbis)==0){
								corrminus = NCC(D.F1, meanI1, corrI1, Point(j+ibis,i+ibis), D.F2, meanI2, corrI2, Point((int)(j+jbis+D.Disparite.at<float>(i,j)-1), i+ibis),(int)blockSize); 
								correqu = NCC(D.F1, meanI1, corrI1, Point(j+jbis,i+ibis), D.F2, meanI2, corrI2, Point((int)(j+jbis+D.Disparite.at<float>(i,j)-1),i+ibis),(int)blockSize); 
								corrplus = NCC(D.F1, meanI1, corrI1, Point(j+jbis,i+ibis), D.F2, meanI2, corrI2, Point((int)(j+jbis+D.Disparite.at<float>(i,j)-1),i+ibis),(int)blockSize);
								if(corrminus>correqu && corrminus>corrplus && corrminus>D.seuilCroissance){
									D.Disparite.at<float>(i+ibis,j+jbis) = D.Disparite.at<float>(i,j)-1;
									D.IdentiteDisp.at<float>(i+ibis,j+jbis) = 1;
									switc = true;
								} else if( correqu>corrminus && correqu >corrplus && correqu>D.seuilCroissance){
									D.Disparite.at<float>(i+ibis,j+jbis) = D.Disparite.at<float>(i,j);
									D.IdentiteDisp.at<float>(i+ibis,j+jbis) = 1;
									switc = true;
								} else if( corrplus>corrminus && corrplus>correqu && corrplus>D.seuilCroissance){
									D.Disparite.at<float>(i+ibis,j+jbis) = D.Disparite.at<float>(i,j)+1;
									D.IdentiteDisp.at<float>(i+ibis,j+jbis) = 1;
									switc = true;
								}
							}
						}
					}
					D.IdentiteDisp.at<float>(i,j)=2;
				}
			}
		}
	}

	cout << "fin de l'affichage de la disparité" << endl;
	
	imshow("Disparité Connue",D.IdentiteDisp);
	imshow("Disparité",D.Disparite);
	
	// On créé désormais l'affichage 3D de la figure

	vector<Point3f> v;
	vector<Vec3i> f;
	vector<Vec3b> col;
	float z;
	for(int i=blockSize+1; i<D.I1.height()-1-blockSize; i++){
		for(int j=blockSize+1; j<D.I1.width()-blockSize-1; j++){
			z = 1000000./(100.+D.Disparite.at<float>(i,j));
			v.push_back(Point3f(j,i,z));
			col.push_back(Vec3b(D.I1.at<uchar>(i,j),D.I1.at<uchar>(i,j),D.I1.at<uchar>(i,j)));
		}
	}
	int largeur = D.I1.height()-2-blockSize;
	for(int i=0; i<D.I1.height()-2-2*blockSize; i++){
		for(int j=0; j<D.I1.width()-2-2*blockSize; j++){
			f.push_back(Vec3i(largeur*i+j, largeur*i+j+1, largeur*(i+1)+j+1));
			f.push_back(Vec3i(largeur*i+j, largeur*(i+1)+j, largeur*(i+1)+j+1));
		}
	}
	savePly("image3D.ply", v, f, col);

	waitKey();
	
	return 0;
}

