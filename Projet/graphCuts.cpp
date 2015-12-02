#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <math.h>  

#include "maxflow/graph.h"
#include "image.h"

using namespace std;
using namespace cv;




struct Data{
	Image<Vec3b> I1, I2;
	Image<float> F1, F2;
	Mat Disparite;
};

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
	cout << "vsize " << v.size() << endl;
	
	return true;
}

int main() {
	//début du programme
	cout << endl << endl << "Programme de Stereo-vision via Graph-cuts" << endl << "Kevin Bienvenu" << endl;
	cout << endl << endl << "initialisation des matrices de travail" << endl;


	// On initialise la structure D qui contient les images intermédiaires et la matrice finale contenant les disparites.
	Data D;
	D.I1 = imread("../Dessin001.jpg");
	D.I2 = imread("../Dessin002.jpg");
	assert(D.I1.height()==D.I2.height());
	//imshow("I1",D.I1);
	imshow("I2",D.I2);


	// On définit les limites de recherche pour la disparite
	// ainsi que le blocksize pour la correlation
	int chercherDispMin = 0;
	int chercherDispMax = 20;
	int diffDisp = chercherDispMax - chercherDispMin;
	int blockSize = 5;

	//remarques: Ces limites de disparités sont les paramètres importants à régler.
	// 1. Plus l'écart est élevé, plus le résultat final sera précis.
	// 2. Un trop grand écart créée trop de sommets dans le graphe qui a une certaine tolérance: au delà, le programme plante.
	// 3. Pour améliorer le programme, on pourrait ré-utiliser le contenu du TD 4 pour détecter automatiquement l'écart.
	// (cf. à la fin)

	// On convertit les images pour les rendre en float
	Image<uchar>G1,G2;
	cvtColor(D.I1,G1,CV_BGR2GRAY);
	cvtColor(D.I2,G2,CV_BGR2GRAY);
	G1.convertTo(D.F1,CV_32F);
	G2.convertTo(D.F2,CV_32F);
	D.Disparite = Mat::zeros(D.I1.rows, D.I1.cols, CV_32F);


	cout << "   fin des declarations des matrices" << endl << endl;
	
	
	// On calcule les matrices intermédiaires qui nous serviront à simplifier les calculs redondants de NCC
	
	Image<float> meanI1 = meanImage(D.F1,(int)blockSize);
	Image<float> meanI2 = meanImage(D.F2,(int)blockSize);
	Image<float> corrI1 = corrImage(D.F1,meanI1,(int)blockSize);
	Image<float> corrI2 = corrImage(D.F2,meanI2,(int)blockSize);
	

	cout << "   => fin des initialisations des matrices intermédiaires" << endl;
	
	cout << endl << endl << "creation du graphe oriente" << endl;
	
	
	int channels = D.F1.channels();
	// ce paramètre est important, car il permet de traiter les images colorées.
	int m = D.F1.rows;
	int n = D.F1.cols;
	long nombreSommet = n*m*diffDisp;
	long nombreArrete = 4*n*m*diffDisp ;

	cout << "   graphe de " << nombreSommet << " sommets et " << nombreArrete << " arretes"<< endl;
	cout << "   creation du graphe" << endl;
	Graph<int,int,int> g(nombreSommet, nombreArrete);
	g.add_node(nombreSommet);
	
	
	// Construisons alors le graph dont nous chercherons l'énergie minimale
	// Pour chaque pixel de l'image, nous calculons la disparite avec le point situé entre chercheDispMin et chercheDispMax
	// Les arrêtes sont créées entre les points en fonction de cette disparité.
	//
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
	//g.add_node(2); 
	//g.add_tweights( 0,   capacities   1, 5 );
	//g.add_tweights( 1,   capacities  6, 1 );
	//g.add_edge( 0, 1,    capacities   4, 3 );
	
	
	float disp;
	Point p1;
	Point p2;
	int capaciteInfinie = 10*diffDisp;
	cout << "    definition de la capacite infinie =  " << capaciteInfinie << endl;

	
	cout << "   creation des arretes" << endl;
	//création des arrêtes vers la source
	for(int i=0; i<n*m; i++){
		g.add_tweights( i*diffDisp, capaciteInfinie, 0 );
	}
	//création des arrêtes verticales et horizontales
	int numeroNoeud = 0, numeroInt, numeroInt2;
	int dispMoy = 0;
	int compteurdebug = 0;
	for(int i=0; i<m; i++){
		//cout << i << " " << endl;
		for(int j=0; j<n; j++){
			numeroInt = numeroNoeud;
			for(int k=chercherDispMin; k<chercherDispMax; k++){
				p1.x = j;
				p1.y = i;
				p2.x = j+k;
				p2.y = i;
				if(j+k-blockSize>=0 && j+k+blockSize<n && i-blockSize>=0 && i+blockSize<m){ 
					disp = 0.;
					disp = NCC(D.F1, meanI1, corrI1, p1, D.F2, meanI2, corrI2, p2,blockSize); 
					//if(i>5)
						//cout << disp << endl;
					disp = 15.-10.*disp;
				} else {
					disp = capaciteInfinie;
				}
				dispMoy += disp;
				//arrêtes selon l'axe puit-source (vertical)
				if(k==chercherDispMax-1){
					g.add_tweights(numeroNoeud,0,(int)disp);
				} else {
					g.add_edge(numeroNoeud,numeroNoeud+1,(int)disp,0);
				}
				numeroNoeud += 1;
			}
			dispMoy = dispMoy / diffDisp;
			for(int k=chercherDispMin; k<chercherDispMax; k++){
				//arrêtes selon l'axe transverse (horizontal)
				if(j<n-1)
					g.add_edge(numeroInt,numeroInt + diffDisp,dispMoy/4,dispMoy/4);
				if(i<m-1 && numeroInt + n *diffDisp>nombreSommet){
					compteurdebug++;
				}
				if(i<m-1 && j<n-1){
					numeroInt2 = numeroInt + n*diffDisp;
					g.add_edge(numeroInt, numeroInt2 ,dispMoy/4,dispMoy/4);
				}
				numeroInt++;
			}
			//cout << i << " " << j << " disp moy= " << dispMoy << endl;
			dispMoy = 0;
		}
		//cout << "i: " << i << " NCC = " << disp << endl;
	}
	//debug
	//cout << "bugs dus a des vertex inexistants " << compteurdebug<< endl;
	
	cout << "   => creation des arretes terminee" << endl;


	//calcul du flot

	cout << endl << "Debut du calcul du flot maximal" << endl;

	int flow = g.maxflow();

	cout << "   flow = " << flow << endl;

	numeroNoeud = 0;
	for (int i=0;i<m;i++){
		for(int j=0; j<n; j++){
			for(int k=chercherDispMin; k<chercherDispMax; k++){
				if (g.what_segment(numeroNoeud) == Graph<int,int,int>::SOURCE){
					// tant que le noeuf appartient à la source, on augmente la disparite.
					D.Disparite.at<float>(i,j)=(float)k;
				}
				numeroNoeud+=1;
			}
			//cout << i << " " << j << " " << D.Disparite.at<float>(i,j) << endl;
		}
	}
	// on a ainsi stockée dans une matrice les résultats optimaux de disparité.

	cout << "   => fin du calcul du flot" << endl;
	
	//imshow("Disparité",D.Disparite);
	
	// On créé désormais l'affichage 3D de la figure

	cout << endl << endl << "Generation de l'image 3D" << endl;

	vector<Point3f> v;
	vector<Vec3i> f;
	vector<Vec3b> col;
	float z;
	//création des sommets et des couleurs.
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			if(i<blockSize || i>m-blockSize ||j<blockSize || j>n-blockSize)
				z = -100000./(100.+D.Disparite.at<float>(blockSize+1,blockSize+1));
			else
				z = -100000./(100.+D.Disparite.at<float>(i,j));
			v.push_back(Point3f(i,j,z));
			col.push_back(D.I1(j,i));
		}
	}
	// création des faces
	for(int i = blockSize ; i < m - blockSize - 1 ; i++){
        for(int j = blockSize ; j < n - blockSize - 1 ; j++){
            f.push_back(Vec3i(j+i*n , j+i*n+1 , j+(i+1)*n+1));
            f.push_back(Vec3i(j+i*n , j+(i+1)*n, 1+j+(i+1)*n));
			if(1+j+(i+1)*n>=v.size())
				cout << "bug " << i << " " << j << endl;
        }
	}

	savePly("image3D.ply", v, f, col);

	cout << "  => fin de la génération de l'image"<< endl;

	
	/* 

	Détaillons à présent le fonctionnement de l'arbre.

	Dans le calcul du maxflow, on cherche les coupe diminuant l'énergie, donc coupant les branches de
	capacités minimales. Plus la corrélation augmente, plus le résultat est satisfaisant, il convient donc
	de ne pas directement mettre la corrélation mais une fonction de celle-ci. J'ai choisi une fonction linéaire
	inverse, car elle satisfait la condition précédente et donne des capacités suffisament faible pour ne
	pas faire exploser le calcul du flow.

	Au début de la construction du graphe il est nécessaire de définir une capacité dite infinie, qui doit
	etre numériquement finie. On la définit de telle sorte qu'elle soit suffisamment haute pour ne pas etre coupée
	mais faible pour ne pas rendre le flow trop grand en cas de calcul.

	Enfin pour éviter les discontinuités on utilise des branches transverse avec une capacité lambda. 
	Dans les cas courants, le lambda est défini à l'avance et constant tout le long du graphe.
	Pour atténuer ici le fait que lambda vaire en fonction de l'image, on décide
	ici de mettre un lambda qui dépend des valeurs dans le graphe, de cette manière, il s'auto-adapte aux
	valeurs de disparités.

	Comme noté plus haut, il est possible d'améliorer le programme.
	
	Dans un premier temps, en combinant le programme avec le contenu du TD4, on peut détecter des graines
	via la méthode de Horner et calculer des disparités dont on est sur de la valeur. À partir de cet
	échantillon on pourrait déduire un écart type de disparité. 

	Néanmoins cette méthode ne prend pas en compte les capacités en mémoire de la machine nécéssaire pour 
	faire tourner le programme du maxflow. En effet, il est impossible de construire un graphe trop grand
	sans obtenir de bug au lancement. Autrement dit, selon la taille de l'image, il faut faire attention à
	réduire la taille de l'écart de disparité recherché.

	
	*/
	
	return 0;
	
}





     