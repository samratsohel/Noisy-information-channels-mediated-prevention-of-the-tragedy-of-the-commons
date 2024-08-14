////////////////////////////////////////////////////////////////////////
// Author: Samrat Sohel Mondal
// Place: IIT Kanpur
// Date: March 30, 2024
////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////
// calling various libraries, packages, and namespaces used in this code
////////////////////////////////////////////////////////////////////////
#include <ctime>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <typeinfo>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
////////////////////////////////////////////////////////////////////////



// random integer number generator //range : [min number to max number]
////////////////////////////////////////////////////////////////////////
int randInt(int min, int max) 
{
	//srand( time(NULL) );
	int z=min + rand() % (( max + 1 ) - min);
	return z;
}
////////////////////////////////////////////////////////////////////////



// rounds float to the fifth decimal place function
////////////////////////////////////////////////////////////////////////
double round(double var) 
{
	double value = (int)(var * 100000 + .5);
	return (double)value / 100000; 
}
////////////////////////////////////////////////////////////////////////



// genration of strategies of different classes {1,2,3,4}
////////////////////////////////////////////////////////////////////////
void strategyGen(int strategyClass,int n,VectorXd & strategy) 
{
	// a strategy is represented using 8-tuple irrespective of the class
    VectorXd binaryNum = VectorXd::Constant(8,0);
    // 1: memory-half does not consider stage of SG
	if (strategyClass == 1)
    {
    	if (n==0){binaryNum<<0,0,0,0,0,0,0,0;}
    	if (n==1){binaryNum<<0,1,0,1,0,1,0,1;}
    	if (n==2){binaryNum<<1,0,1,0,1,0,1,0;}
    	if (n==3){binaryNum<<1,1,1,1,1,1,1,1;}
    }
    // 2: memory-half considers stage of SG
    if (strategyClass == 2)
    {
    	int D = 4;
    	int j = D-1; 
    	while (n > 0){
    		binaryNum(j) = n % 2;
    		n = n / 2; 
    		j--; 
    	}
    	binaryNum(5) = binaryNum(3);
    	binaryNum(7) = binaryNum(3);
    	binaryNum(4) = binaryNum(2);
    	binaryNum(6) = binaryNum(2);
    	binaryNum(3) = binaryNum(1);
    	binaryNum(2) = binaryNum(0);
    }
    // 3: memery-one does not consider stage of SG
    if (strategyClass == 3)
    {
    	int D = 4;
    	int j = D-1; 
    	while (n > 0){
    		binaryNum(j) = n % 2;
    		n = n / 2; 
    		j--; 
    	}
    	for (int j = 0; j < 4; ++j)
    	{
    		binaryNum(j+4) = binaryNum(j);
    	}
    }
	// 4: memory-one considers stage of SG
    if (strategyClass == 4)
    {
    	int D = 8;
    	int j = D-1; 
    	while (n > 0){
    		binaryNum(j) = n % 2;
    		n = n / 2; 
    		j--; 
    	}
    }
    strategy = binaryNum;
}
////////////////////////////////////////////////////////////////////////



// size of a strategy class generation
////////////////////////////////////////////////////////////////////////
void classSizeGen(int strategyClass,int & classSize) 
{
	if (strategyClass == 1){classSize =   4;}
	if (strategyClass == 2){classSize =  16;}
	if (strategyClass == 3){classSize =  16;}
	if (strategyClass == 4){classSize = 256;}
}
////////////////////////////////////////////////////////////////////////



// q vector generation
// input: (q vector number \in {0,1,2})
// output: (one of the three important transition vectors)
////////////////////////////////////////////////////////////////////////
void qvecGen(int qvecNo, VectorXd & qvec)
{
	VectorXd q(8);
	if (qvecNo == 1){q << 1,0,0,0,1,0,0,0;}
	if (qvecNo == 2){q << 1,0,0,0,1,1,1,1;}
	if (qvecNo == 3){q << 1,0,0,0,1,1,1,0;}
	qvec = q;
}
////////////////////////////////////////////////////////////////////////



// payoff and cooperation given a distribution 
// input: (equilibrium distrbution over states)
// output: (payoff1, payoff2, cooperation1, cooperation2)
////////////////////////////////////////////////////////////////////////
void payCoop(VectorXd V,double &payoff1,double &payoff2,double &coop1,double &coop2,double &stage1freq)
{
	// values of the entries of payoff matrix
	double b1=2.0,   b2=1.2, c=1;
	double R1=b1-c,  S1=-c,  T1=b1,  P1=0;
	double R2=b2-c,  S2=-c,  T2=b2,  P2=0;
	VectorXd payoff_vec1(8);payoff_vec1<<R1,S1,T1,P1,R2,S2,T2,P2;
	VectorXd payoff_vec2(8);payoff_vec2<<R1,T1,S1,P1,R2,T2,S2,P2;
	payoff1=payoff_vec1.dot(V);coop1=(V(0)+V(1)+V(4)+V(5));
	payoff2=payoff_vec2.dot(V);coop2=(V(0)+V(2)+V(4)+V(6));
	stage1freq = (V(0)+V(1)+V(2)+V(3));
}
////////////////////////////////////////////////////////////////////////



// transformation of a stratgey beacuse of information channel
// input: (error and information channels [error rate: [epsC,epsD], first channel: [n1,n2], second channel: [wC,wD]]) and (strategy)
// output: (effective strategy)
////////////////////////////////////////////////////////////////////////
void effectiveStrategyGen(VectorXd errorChannels,VectorXd &p){
	double epsC = errorChannels(0);
	double epsD = errorChannels(1);
	double   n1 = errorChannels(2);
	double   n2 = errorChannels(3);
	double   wC = errorChannels(4);
	double   wD = errorChannels(5);

	VectorXd pUc(8);   pUc << p(4),p(5),p(6),p(7),p(0),p(1),p(2),p(3);
	VectorXd pDc(8);   pDc << p(1),p(0),p(3),p(2),p(5),p(4),p(7),p(6);
	VectorXd pUDc(8); pUDc << p(5),p(4),p(7),p(6),p(1),p(0),p(3),p(2);
	VectorXd ni(8);     ni << n1,n1,n1,n1,n2,n2,n2,n2;
	VectorXd wa(8);     wa << wC,wD,wC,wD,wC,wD,wC,wD;
	VectorXd wac(8);   wac << wD,wC,wD,wC,wD,wC,wD,wC;

	VectorXd pT(8);
	for (int i = 0; i < 8; ++i)
	{
		pT(i) = epsD+(1- epsC-epsD)*((1-ni(i))*(1-wa(i))*p(i)+(1-ni(i))*wa(i)*pDc(i)+ni(i)*(1-wac(i))*pUc(i)+ni(i)*wac(i)*pUDc(i));
	}
	p = pT;
}
////////////////////////////////////////////////////////////////////////



// payoff and cooperation genration given two strategy number
// input: (strategy class), (error and information channels), (focal strategy no.), (opponent strategy no.), (q vector no.), moves scheme \in {1: simultaneous, 2: alternating}
// output: (expected payoff1, payoff2, cooperation1, cooperation2)
////////////////////////////////////////////////////////////////////////
void payCoopGen(int strategyClass,VectorXd errorChannels,int strategyNo1,int strategyNo2,int qvecNo,int scheme,VectorXd &P12C12S1)
{
	// the transition vector is generated
	VectorXd qvec(8);
	qvecGen(qvecNo,qvec);

	// pure strategy pair is generated
	VectorXd p(8),q(8);
	strategyGen(strategyClass,strategyNo1,p);
	strategyGen(strategyClass,strategyNo2,q);

	// effect of error and infromation channels are included
	effectiveStrategyGen(errorChannels,p);
	effectiveStrategyGen(errorChannels,q);

	// transition matrix generation
	MatrixXd M(8,8);
	for (int i = 0; i < 8; ++i)
	{
		int initStage = 1;
		if (i>3)
		{
			initStage = 2;
		}
		double focalCinit = 0; // focal player cooperated initially
		if (i == 0 or i == 1 or i == 4 or i == 5)
		{
			focalCinit = 1;
		}
		for (int j = 0; j < 8; ++j)
		{
			int finalStage = 1;
			if (j>3)
			{
				finalStage = 2;
			}

			// calculation of the first factor of the element of the transition matrix
			double x  = 1;
			if (finalStage == 1){x = x*qvec(i);}
			if (finalStage == 2){x = x*(1-qvec(i));}

			// calculation of the second factor of the element of the transition matrix
			int iF = i;
			if (initStage >  finalStage){iF = iF-4;}
			if (initStage <  finalStage){iF = iF+4;}
			double yF = 1-p(iF);
			double focalCfinal = 0; // focal player cooperated finally
			if (j == 0 or j == 1 or j == 4 or j == 5){
				yF = p(iF); focalCfinal = 1;
			}

			// calculation of the second factor of the element of the transition matrix
			int iO = i;
			if (i == 1){iO = 2;}
			if (i == 2){iO = 1;}
			if (i == 5){iO = 6;}
			if (i == 6){iO = 5;}
			if (initStage >  finalStage){iO = iO-4;}
			if (initStage <  finalStage){iO = iO+4;}
			if (scheme == 2)
			{	
				// scheme of alternating moves is applied
				if (focalCinit>focalCfinal){iO = iO+1;}
				if (focalCinit<focalCfinal){iO = iO-1;}
			}
			double yO = 1-q(iO);
			if (j == 0 or j == 2 or j == 4 or j == 6){
				yO = q(iO);
			}

			M(i,j) = x*yF*yO;
		}
		
	}
	// cout<<M<<endl;
	// Transposition of transition matrix
	MatrixXd MT(8,8);
	MT=M.transpose();
	// right eigen vectors of transposition of transition matrix calculation with eiegen-value unity
	EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
	MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
	int evc;
	double maxz = 0.0;
	for (int i = 0; i < 8; ++i)
	{
		// cout<<es.eigenvectors().col(i).real().transpose()<<endl;
		if (z(i) > maxz)
		{
			maxz = z(i);
			evc=i;
		}
	}
	VectorXd V = es.eigenvectors().col(evc).real();
	double SUM = V.sum();V=V/SUM;
	double payoff1,payoff2,coop1,coop2,stage1freq;
	payCoop(V,payoff1,payoff2,coop1,coop2,stage1freq);
	P12C12S1<<payoff1,payoff2,coop1,coop2,stage1freq;
}
////////////////////////////////////////////////////////////////////////



// fixation probability matrix calculation given the total payoff matrix
// input: (classSize \times classSize Payoff matrix), (strategyNo1),(strategyNo2), (populationSize), (selction stregth)
////////////////////////////////////////////////////////////////////////
double calcRho(MatrixXd PayM,int strategyNo1,int strategyNo2,int N,double beta)
{
	double avgpayoff1,avgpayoff2,alpha=0,cumprod=1,cumprodsum=0;
	for (int i = 1; i < N; ++i)
	{
		avgpayoff1=(double(i-1)/double(N-1))*PayM(strategyNo1,strategyNo1)+(double(N-i)/double(N-1))*PayM(strategyNo1,strategyNo2);
		avgpayoff2=(double(i)/double(N-1))*PayM(strategyNo2,strategyNo1)+(double(N-i-1)/double(N-1))*PayM(strategyNo2,strategyNo2);
		alpha=exp(-beta*(avgpayoff1-avgpayoff2));
		cumprod=cumprod*alpha;
		cumprodsum=cumprodsum+cumprod;
	}
	double rho=1.0/(1.0+cumprodsum);
	return rho;
}
////////////////////////////////////////////////////////////////////////



// genrates ensemble averaged trajectories starting from ALLD with time
// input: (scheme of moves), (strategy class), (q vector), (error rate), (n: first channel), (w: second channle), (selection stregth)
// saves a trajectory in local folder
////////////////////////////////////////////////////////////////////////
void symmetricErrorChannelsTrajGen(int scheme,int strategyClass,int qvecNo,double eps,double n,double w,double beta, const std::string& filename)
{
	// parmeters that are less frequently changed
	int nGEN = 5000;
	int N    = 100;

	VectorXd errorChannels(6); 
	errorChannels<<eps,eps,n,n,w,w;
	int classSize;
	classSizeGen(strategyClass,classSize);

	MatrixXd PayM(classSize,classSize);
	MatrixXd CoopM(classSize,classSize);
	MatrixXd RhoM(classSize,classSize);
	MatrixXd S1freqM(classSize,classSize);
	VectorXd P12C12S1(5);

	for (int i = 0; i < classSize; ++i)
	{
		for (int j = i; j < classSize; ++j)
		{
			payCoopGen(strategyClass,errorChannels,i,j,qvecNo,scheme,P12C12S1);
			PayM(i,j)    = P12C12S1(0);PayM(j,i)    = P12C12S1(1);
			CoopM(i,j)   = P12C12S1(2);CoopM(j,i)   = P12C12S1(3);
			S1freqM(i,j) = P12C12S1(4);S1freqM(j,i) = P12C12S1(4);
		}
	}

	// Calculation of Rho matrix
	/////////////////////////////////////////////
	for (int i = 0; i < classSize; ++i)
	{
		for (int j = 0; j < classSize; ++j)
		{
			if (i!=j)
			{
				RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
			}
		}
		RhoM(i,i) = 0;
		double summ = 0;
		for (int j = 0; j < classSize; ++j)
		{
			summ = summ +RhoM(i,j);
		}
		RhoM(i,i)=1.0-summ;
	}
	VectorXd nITcooperation(nGEN);
	VectorXd POP=VectorXd::Constant(classSize,0);
	POP(0)=1;// population vector: initialli all 'ALLD'

	// Time evolution of the population vector
	/////////////////////////////////////////////
	for (int t = 0; t < nGEN; ++t)
	{
		double cAvg = 0;
		for (int i = 0; i < classSize; ++i)
		{
			cAvg += POP(i)*CoopM(i,i);
		}
		nITcooperation(t) = cAvg;
		cout<<cAvg<<endl;
		MatrixXd POP2=POP.transpose()*RhoM;
		POP = POP2.transpose();
	}

	ofstream file1;
	file1.open("t.txt");
	ofstream file2;
	file2.open(filename);
	for (int i = 0; i < nGEN; ++i)
	{
		file1<<i<<endl;
		file2<<nITcooperation(i)<<endl;
	}
	file1.close();
	file2.close();
}
////////////////////////////////////////////////////////////////////////



// entropy function; needed for mutual infroamtion and capacity calculations
// input: any probability distribution ([x,(1-x)]), o'er the inputs of the channel
////////////////////////////////////////////////////////////////////////
double Hentropy(double x){
	double z = x;
	double tolerance = pow(10,-5);
	if (x == 0){z = tolerance;}
	double entropy = -z*log2(z)-(1.0-z)*log2(1.0-z);
	return entropy;
}
////////////////////////////////////////////////////////////////////////



// capacity function; needed to calculate the capacity of given channel
// input: any channel description ([x1,x2])
////////////////////////////////////////////////////////////////////////
double Capacity(double x1,double x2){
	double z1 = x1;
	double z2 = x2;
	double tolerance = pow(10,-5);
	if (x1 == 0){z1 = tolerance;}
	if (x2 == 0){z2 = tolerance;}
	if (round(x1+x2) == 1){z1 -= (tolerance/2.0);z2 -= (tolerance/2.0);}
	double cap = log2(1.0+pow(2,((Hentropy(z1)-Hentropy(z2))/(1.0-z1-z2))))-((1-z2)/(1.0-z1-z2))*Hentropy(z1) + (z1/(1.0-z1-z2))*Hentropy(z2);
	return cap;
}
////////////////////////////////////////////////////////////////////////



// Mutual infromation function; needed to calculate the mutual information of given distribution and channel
// input: any probability distribution ([x,(1-x)]), any channel description ([x1,x2])
////////////////////////////////////////////////////////////////////////
double Mutual(double x,double x1,double x2){
	double z1 = x1;
	double z2 = x2;
	double alphaIn = x;
	double alphaOut = (1.0-z1)*alphaIn+z2*(1.0-alphaIn);
	double tolerance = pow(10,-5);
	if (x1 == 0){z1 = tolerance;}
	if (x2 == 0){z2 = tolerance;}
	if (round(x1+x2) == 1){z1 -= (tolerance/2.0);z2 -= (tolerance/2.0);}
	double mInformation = Hentropy(alphaOut)-alphaIn*Hentropy(z1)-(1.0-alphaIn)*Hentropy(z2);;
	return mInformation;
}
////////////////////////////////////////////////////////////////////////



// genrates outputs for whole (n) cjannels space
// input: (scheme of moves), (strategy class), (q vector), (error rate), (selection stregth)
// saves equilibrium outcomes: coorperation rate, efficacy(first channel), efficay second channel
////////////////////////////////////////////////////////////////////////
void symmetricErrorChannelsSpace(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1, const std::string& filename2)
{
	// parmeters that are less frequently changed and some initializations
	double dx = 0.5/double(100); // elementary grid size in the channels space
	int N = 100;
	double w = 0.0;
	int classSize;
	classSizeGen(strategyClass,classSize);
	ofstream file1;
	file1.open(filename1);
	ofstream file2;
	file2.open(filename2);
	for (double n = 0; n < 0.5+dx; n += dx)
	{
		VectorXd P12C12S1(5);
		VectorXd errorChannels(6);
		MatrixXd PayM(classSize,classSize);
		MatrixXd RhoM(classSize,classSize);
		MatrixXd CoopM(classSize,classSize);
		MatrixXd S1freqM(classSize,classSize);
		errorChannels<<eps,eps,n,n,w,w;
		// PayM CoopM, S1freqM are calcculated for all the startegies
		/////////////////////////////////////////////
		for (int i = 0; i < classSize; ++i)
		{
			for (int j = i; j < classSize; ++j)
			{

				payCoopGen(strategyClass,errorChannels,i,j,qvecNo,scheme,P12C12S1);

				PayM(i,j)    = P12C12S1(0);PayM(j,i)    = P12C12S1(1);
				CoopM(i,j)   = P12C12S1(2);CoopM(j,i)   = P12C12S1(3);
				S1freqM(i,j) = P12C12S1(4);S1freqM(j,i) = P12C12S1(4);
			}
		}
		// Calculation of Rho matrix
		/////////////////////////////////////////////
		for (int i = 0; i < classSize; ++i)
		{
			for (int j = 0; j < classSize; ++j)
			{
				if (i!=j)
				{
					RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
				}
			}
			RhoM(i,i) = 0;
			double summ = 0;
			for (int j = 0; j < classSize; ++j)
			{
				summ = summ +RhoM(i,j);
			}
			RhoM(i,i)=1.0-summ;
		}
		MatrixXd MT(classSize,classSize);
		MT=RhoM.transpose();
		EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
		MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
		int evc;
		double maxz = 0.0;
		for (int i = 0; i < classSize; ++i)
		{
			if (z(i) > maxz)
			{
				maxz = z(i);
				evc=i;
			}
		}
		VectorXd A = es.eigenvectors().col(evc).real();
		double SUM = A.sum();
		A = A/SUM;// relative abundances of all the strategies are computed

		double c = 0;
		double alphaIn = 0;
		for (int i = 0; i < classSize; ++i)
		{
			c += A(i)*CoopM(i,i);
			alphaIn += A(i)*S1freqM(i,i);
		}
		file1<<c<<" "<<endl;
		file2<<alphaIn<<" "<<endl;
		cout<<"=================================="<<endl;
		cout<<"n="<<n<<endl;
		cout<<"c="<<c<<endl;
		cout<<"alphaIn="<<alphaIn<<endl;
		cout<<"=================================="<<endl;
	}
	file1.close();
	file2.close();
}
////////////////////////////////////////////////////////////////////////


// genrates outputs for whole (n) cjannels space
// input: (scheme of moves), (strategy class), (q vector), (error rate), (selection stregth)
// saves equilibrium outcomes: coorperation rate, efficacy(first channel), efficay second channel
////////////////////////////////////////////////////////////////////////
void symmetricErrorChannelsSpaceCT(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1, const std::string& filename2)
{
	// parmeters that are less frequently changed and some initializations
	double dx = 0.5/double(100); // elementary grid size in the channels space
	double w = 0.0;
	int classSize;
	classSizeGen(strategyClass,classSize);
	ofstream file1;
	file1.open(filename1);
	ofstream file2;
	file2.open(filename2);
	double n = 0.5;
	for (int N = 0; N < 200+1; N += 1)
	{
		VectorXd P12C12S1(5);
		VectorXd errorChannels(6);
		MatrixXd PayM(classSize,classSize);
		MatrixXd RhoM(classSize,classSize);
		MatrixXd CoopM(classSize,classSize);
		MatrixXd S1freqM(classSize,classSize);
		errorChannels<<eps,eps,n,n,w,w;
		// PayM CoopM, S1freqM are calcculated for all the startegies
		/////////////////////////////////////////////
		for (int i = 0; i < classSize; ++i)
		{
			for (int j = i; j < classSize; ++j)
			{

				payCoopGen(strategyClass,errorChannels,i,j,qvecNo,scheme,P12C12S1);

				PayM(i,j)    = P12C12S1(0);PayM(j,i)    = P12C12S1(1);
				CoopM(i,j)   = P12C12S1(2);CoopM(j,i)   = P12C12S1(3);
				S1freqM(i,j) = P12C12S1(4);S1freqM(j,i) = P12C12S1(4);
			}
		}
		// Calculation of Rho matrix
		/////////////////////////////////////////////
		for (int i = 0; i < classSize; ++i)
		{
			for (int j = 0; j < classSize; ++j)
			{
				if (i!=j)
				{
					RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
				}
			}
			RhoM(i,i) = 0;
			double summ = 0;
			for (int j = 0; j < classSize; ++j)
			{
				summ = summ +RhoM(i,j);
			}
			RhoM(i,i)=1.0-summ;
		}
		MatrixXd MT(classSize,classSize);
		MT=RhoM.transpose();
		EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
		MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
		int evc;
		double maxz = 0.0;
		for (int i = 0; i < classSize; ++i)
		{
			if (z(i) > maxz)
			{
				maxz = z(i);
				evc=i;
			}
		}
		VectorXd A = es.eigenvectors().col(evc).real();
		double SUM = A.sum();
		A = A/SUM;// relative abundances of all the strategies are computed

		double c = 0;
		double alphaIn = 0;
		for (int i = 0; i < classSize; ++i)
		{
			c += A(i)*CoopM(i,i);
			alphaIn += A(i)*S1freqM(i,i);
		}
		file1<<c<<" "<<endl;
		file2<<alphaIn<<" "<<endl;
		cout<<"=================================="<<endl;
		cout<<"N="<<N<<endl;
		cout<<"c="<<c<<endl;
		cout<<"alphaIn="<<alphaIn<<endl;
		cout<<"=================================="<<endl;
	}
	file1.close();
	file2.close();
}
////////////////////////////////////////////////////////////////////////


// genrates outputs for whole (n) cjannels space
// input: (scheme of moves), (strategy class), (q vector), (error rate), (selection stregth)
// saves equilibrium outcomes: coorperation rate, efficacy(first channel), efficay second channel
////////////////////////////////////////////////////////////////////////
void symmetricErrorChannelsSpaceW(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1, const std::string& filename2)
{
	// parmeters that are less frequently changed and some initializations
	double dx = 0.5/double(100); // elementary grid size in the channels space
	int N = 100;
	double n = 0.0;
	int classSize;
	classSizeGen(strategyClass,classSize);
	ofstream file1;
	file1.open(filename1);
	ofstream file2;
	file2.open(filename2);
	for (double w = 0; w < 0.5+dx; w += dx)
	{
		VectorXd P12C12S1(5);
		VectorXd errorChannels(6);
		MatrixXd PayM(classSize,classSize);
		MatrixXd RhoM(classSize,classSize);
		MatrixXd CoopM(classSize,classSize);
		MatrixXd S1freqM(classSize,classSize);
		errorChannels<<eps,eps,n,n,w,w;
		// PayM CoopM, S1freqM are calcculated for all the startegies
		/////////////////////////////////////////////
		for (int i = 0; i < classSize; ++i)
		{
			for (int j = i; j < classSize; ++j)
			{

				payCoopGen(strategyClass,errorChannels,i,j,qvecNo,scheme,P12C12S1);

				PayM(i,j)    = P12C12S1(0);PayM(j,i)    = P12C12S1(1);
				CoopM(i,j)   = P12C12S1(2);CoopM(j,i)   = P12C12S1(3);
				S1freqM(i,j) = P12C12S1(4);S1freqM(j,i) = P12C12S1(4);
			}
		}
		// Calculation of Rho matrix
		/////////////////////////////////////////////
		for (int i = 0; i < classSize; ++i)
		{
			for (int j = 0; j < classSize; ++j)
			{
				if (i!=j)
				{
					RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
				}
			}
			RhoM(i,i) = 0;
			double summ = 0;
			for (int j = 0; j < classSize; ++j)
			{
				summ = summ +RhoM(i,j);
			}
			RhoM(i,i)=1.0-summ;
		}
		MatrixXd MT(classSize,classSize);
		MT=RhoM.transpose();
		EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
		MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
		int evc;
		double maxz = 0.0;
		for (int i = 0; i < classSize; ++i)
		{
			if (z(i) > maxz)
			{
				maxz = z(i);
				evc=i;
			}
		}
		VectorXd A = es.eigenvectors().col(evc).real();
		double SUM = A.sum();
		A = A/SUM;// relative abundances of all the strategies are computed

		double c = 0;
		double alphaIn = 0;
		for (int i = 0; i < classSize; ++i)
		{
			c += A(i)*CoopM(i,i);
			alphaIn += A(i)*S1freqM(i,i);
		}
		file1<<c<<" "<<endl;
		file2<<alphaIn<<" "<<endl;
		cout<<"=================================="<<endl;
		cout<<"w="<<w<<endl;
		cout<<"c="<<c<<endl;
		cout<<"alphaIn="<<alphaIn<<endl;
		cout<<"=================================="<<endl;
	}
	file1.close();
	file2.close();
}
////////////////////////////////////////////////////////////////////////



// genrates outputs for whole (n1-n2) cjannels space
// input: (scheme of moves), (strategy class), (q vector), (error rate), (selection stregth)
// saves equilibrium outcomes: coorperation rate, efficacy(first channel), efficay second channel
////////////////////////////////////////////////////////////////////////
void AsymmetricErrorChannelsSpace(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1,const std::string& filename2,const std::string& filename3,const std::string& filename4,const std::string& filename5)
{
	// parmeters that are less frequently changed and some initializations
	double dx = 1.0/double(100); // elementary grid size in the channels space
	int N = 100;
	double w = 0.0;
	int classSize;
	classSizeGen(strategyClass,classSize);
	ofstream file1;
	file1.open(filename1);
	ofstream file2;
	file2.open(filename2);
	ofstream file3;
	file3.open(filename3);
	ofstream file4;
	file4.open(filename4);
	ofstream file5;
	file5.open(filename5);
	for (double n1 = 0; n1 < 1.0+dx; n1 += dx)
	{
		for (double n2 = 0; n2 < 1.0+dx; n2 += dx)
		{
			double c = 0;
			double alphaIn = 0;
			double mInformationN = 0;
			double capacityN = 1.0;
			int abundant1 = 15;
			double percent1 = 0;
			if (n1+n2 <= 1+dx)
			{
				VectorXd P12C12S1(5);
				VectorXd errorChannels(6);
				MatrixXd PayM(classSize,classSize);
				MatrixXd RhoM(classSize,classSize);
				MatrixXd CoopM(classSize,classSize);
				MatrixXd S1freqM(classSize,classSize);
				errorChannels<<eps,eps,n1,n2,w,w;

				// PayM CoopM, S1freqM are calcculated for all the startegies
				/////////////////////////////////////////////
				for (int i = 0; i < classSize; ++i)
				{
					for (int j = i; j < classSize; ++j)
					{

						payCoopGen(strategyClass,errorChannels,i,j,qvecNo,scheme,P12C12S1);

						PayM(i,j)    = P12C12S1(0);PayM(j,i)    = P12C12S1(1);
						CoopM(i,j)   = P12C12S1(2);CoopM(j,i)   = P12C12S1(3);
						S1freqM(i,j) = P12C12S1(4);S1freqM(j,i) = P12C12S1(4);
					}
				}
				// Calculation of Rho matrix
				/////////////////////////////////////////////
				for (int i = 0; i < classSize; ++i)
				{
					for (int j = 0; j < classSize; ++j)
					{
						if (i!=j)
						{
							RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
						}
					}
					RhoM(i,i) = 0;
					double summ = 0;
					for (int j = 0; j < classSize; ++j)
					{
						summ = summ +RhoM(i,j);
					}
					RhoM(i,i)=1.0-summ;
				}
				MatrixXd MT(classSize,classSize);
				MT=RhoM.transpose();
				EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
				MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
				int evc;
				double maxz = 0.0;
				for (int i = 0; i < classSize; ++i)
				{
					if (z(i) > maxz)
					{
						maxz = z(i);
						evc=i;
					}
				}
				VectorXd A = es.eigenvectors().col(evc).real();
				double SUM = A.sum();
				A = A/SUM;// relative abundances of all the strategies are computed

				for (int i = 0; i < classSize; ++i)
				{
					c += A(i)*CoopM(i,i);
					alphaIn = S1freqM(i,i);
					mInformationN += A(i)*Mutual(alphaIn,n1,n2);
					if (A(i)>=A(abundant1))
					{
						abundant1 = i;
					}
				}
				capacityN = Capacity(n1,n2);
				percent1 = A(abundant1);
			}
			file1<<c<<" ";
			file2<<mInformationN<<" ";
			file3<<capacityN<<" ";
			file4<<abundant1<<" ";
			file5<<percent1<<" ";
			cout<<"=================================="<<endl;
			cout<<"n1="<<n1<<" "<<"n2="<<n2<<endl;
			cout<<"=================================="<<endl;
		}
		file1<<endl;
		file2<<endl;
		file3<<endl;
		file4<<endl;
		file5<<endl;
	}
	file1.close();
	file2.close();
	file3.close();
	file4.close();
	file5.close();
}
////////////////////////////////////////////////////////////////////////



// genrates outputs for whole (w1-w2) cjannels space
// input: (scheme of moves), (strategy class), (q vector), (error rate), (selection stregth)
// saves equilibrium outcomes: coorperation rate, efficacy(first channel), efficay second channel
////////////////////////////////////////////////////////////////////////
void AsymmetricErrorChannelsSpaceW(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1,const std::string& filename2,const std::string& filename3,const std::string& filename4,const std::string& filename5)
{
	// parmeters that are less frequently changed and some initializations
	double dx = 1.0/double(100); // elementary grid size in the channels space
	int N = 100;
	double n = 0.0;
	int classSize;
	classSizeGen(strategyClass,classSize);
	ofstream file1;
	file1.open(filename1);
	ofstream file2;
	file2.open(filename2);
	ofstream file3;
	file3.open(filename3);
	ofstream file4;
	file4.open(filename4);
	ofstream file5;
	file5.open(filename5);
	for (double w1 = 0; w1 < 1.0+dx; w1 += dx)
	{
		for (double w2 = 0; w2 < 1.0+dx; w2 += dx)
		{
			double c = 0;
			double alphaIn = 0;
			double mInformationW = 0;
			double capacityW = 1.0;
			int abundant1 = 15;
			double percent1 = 0;
			if (w1+w2 <= 1+dx)
			{
				VectorXd P12C12S1(5);
				VectorXd errorChannels(6);
				MatrixXd PayM(classSize,classSize);
				MatrixXd RhoM(classSize,classSize);
				MatrixXd CoopM(classSize,classSize);
				MatrixXd S1freqM(classSize,classSize);
				errorChannels<<eps,eps,n,n,w1,w2;

				// PayM CoopM, S1freqM are calcculated for all the startegies
				/////////////////////////////////////////////
				for (int i = 0; i < classSize; ++i)
				{
					for (int j = i; j < classSize; ++j)
					{

						payCoopGen(strategyClass,errorChannels,i,j,qvecNo,scheme,P12C12S1);

						PayM(i,j)    = P12C12S1(0);PayM(j,i)    = P12C12S1(1);
						CoopM(i,j)   = P12C12S1(2);CoopM(j,i)   = P12C12S1(3);
						S1freqM(i,j) = P12C12S1(4);S1freqM(j,i) = P12C12S1(4);
					}
				}
				// Calculation of Rho matrix
				/////////////////////////////////////////////
				for (int i = 0; i < classSize; ++i)
				{
					for (int j = 0; j < classSize; ++j)
					{
						if (i!=j)
						{
							RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
						}
					}
					RhoM(i,i) = 0;
					double summ = 0;
					for (int j = 0; j < classSize; ++j)
					{
						summ = summ +RhoM(i,j);
					}
					RhoM(i,i)=1.0-summ;
				}
				MatrixXd MT(classSize,classSize);
				MT=RhoM.transpose();
				EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
				MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
				int evc;
				double maxz = 0.0;
				for (int i = 0; i < classSize; ++i)
				{
					if (z(i) > maxz)
					{
						maxz = z(i);
						evc=i;
					}
				}
				VectorXd A = es.eigenvectors().col(evc).real();
				double SUM = A.sum();
				A = A/SUM;// relative abundances of all the strategies are computed

				for (int i = 0; i < classSize; ++i)
				{
					c += A(i)*CoopM(i,i);
					alphaIn = CoopM(i,i);
					mInformationW += A(i)*Mutual(alphaIn,w1,w2);
					if (A(i)>=A(abundant1))
					{
						abundant1 = i;
					}
				}
				capacityW = Capacity(w1,w2);
				percent1 = A(abundant1);
			}
			file1<<c<<" ";
			file2<<mInformationW<<" ";
			file3<<capacityW<<" ";
			file4<<abundant1<<" ";
			file5<<percent1<<" ";
			cout<<"=================================="<<endl;
			cout<<"w1="<<w1<<" "<<"w2="<<w2<<endl;
			cout<<"=================================="<<endl;
		}
		file1<<endl;
		file2<<endl;
		file3<<endl;
		file4<<endl;
		file5<<endl;
	}
	file1.close();
	file2.close();
	file3.close();
	file4.close();
	file5.close();
}
////////////////////////////////////////////////////////////////////////



// genrates outputs for whole (n-w) cjannels space
// input: (scheme of moves), (strategy class), (q vector), (error rate), (selection stregth)
// saves equilibrium outcomes: coorperation rate, efficacy(first channel), efficay second channel
////////////////////////////////////////////////////////////////////////
void tMatAbundanceSave(int scheme,int strategyClass,int qvecNo,double eps,double beta,double n1,double n2,const std::string& filename1,const std::string& filename2)
{
	// parmeters that are less frequently changed and some initializations
	int N = 100;
	double w = 0.0;
	int classSize;
	classSizeGen(strategyClass,classSize);
	ofstream file1;
	file1.open(filename1);
	ofstream file2;
	file2.open(filename2);
	VectorXd P12C12S1(5);
	VectorXd errorChannels(6);
	MatrixXd PayM(classSize,classSize);
	MatrixXd RhoM(classSize,classSize);
	MatrixXd CoopM(classSize,classSize);
	MatrixXd S1freqM(classSize,classSize);
	errorChannels<<eps,eps,n1,n2,w,w;

	// PayM CoopM, S1freqM are calcculated for all the startegies
	/////////////////////////////////////////////
	for (int i = 0; i < classSize; ++i)
	{
		for (int j = i; j < classSize; ++j)
		{

			payCoopGen(strategyClass,errorChannels,i,j,qvecNo,scheme,P12C12S1);

			PayM(i,j)    = P12C12S1(0);PayM(j,i)    = P12C12S1(1);
			CoopM(i,j)   = P12C12S1(2);CoopM(j,i)   = P12C12S1(3);
			S1freqM(i,j) = P12C12S1(4);S1freqM(j,i) = P12C12S1(4);
		}
	}
	// Calculation of Rho matrix
	/////////////////////////////////////////////
	for (int i = 0; i < classSize; ++i)
	{
		for (int j = 0; j < classSize; ++j)
		{
			if (i!=j)
			{
				RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
			}
		}
		RhoM(i,i) = 0;
		double summ = 0;
		for (int j = 0; j < classSize; ++j)
		{
			summ = summ +RhoM(i,j);
		}
		RhoM(i,i)=1.0-summ;
	}
	MatrixXd MT(classSize,classSize);
	MT=RhoM.transpose();
	EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
	MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
	int evc;
	double maxz = 0.0;
	for (int i = 0; i < classSize; ++i)
	{
		if (z(i) > maxz)
		{
			maxz = z(i);
			evc=i;
		}
	}

	VectorXd A = es.eigenvectors().col(evc).real();
	double SUM = A.sum();
	A = A/SUM;// relative abundances of all the strategies are computed

	for (int i = 0; i < classSize; ++i)
	{
		for (int j = 0; j < classSize; ++j)
		{
			file1<<RhoM(i,j)<<" ";
			
		}
		file1<<endl;
		file2<<A(i)<<endl;
	}
	file1.close();
	file2.close();
}
////////////////////////////////////////////////////////////////////////



// payoff and cooperation given a distribution 
// input: (equilibrium distrbution over states)
// output: (payoff1, payoff2, cooperation1, cooperation2,b1)
////////////////////////////////////////////////////////////////////////
void payCoopb1(VectorXd V,double &payoff1,double &payoff2,double &coop1,double &coop2,double &stage1freq, double b1)
{
	// values of the entries of payoff matrix
	double b2=1.2, c=1;
	double R1=b1-c,  S1=-c,  T1=b1,  P1=0;
	double R2=b2-c,  S2=-c,  T2=b2,  P2=0;
	VectorXd payoff_vec1(8);payoff_vec1<<R1,S1,T1,P1,R2,S2,T2,P2;
	VectorXd payoff_vec2(8);payoff_vec2<<R1,T1,S1,P1,R2,T2,S2,P2;
	payoff1=payoff_vec1.dot(V);coop1=(V(0)+V(1)+V(4)+V(5));
	payoff2=payoff_vec2.dot(V);coop2=(V(0)+V(2)+V(4)+V(6));
	stage1freq = (V(0)+V(1)+V(2)+V(3));
}
////////////////////////////////////////////////////////////////////////



// payoff and cooperation genration given two strategy number
// input: (strategy class), (error and information channels), (focal strategy no.), (opponent strategy no.), (q vector no.), moves scheme \in {1: simultaneous, 2: alternating}
// output: (expected payoff1, payoff2, cooperation1, cooperation2,b1)
////////////////////////////////////////////////////////////////////////
void payCoopGenb1(int strategyClass,VectorXd errorChannels,int strategyNo1,int strategyNo2,int qvecNo,int scheme,VectorXd &P12C12S1, double b1)
{
	// the transition vector is generated
	VectorXd qvec(8);
	qvecGen(qvecNo,qvec);

	// pure strategy pair is generated
	VectorXd p(8),q(8);
	strategyGen(strategyClass,strategyNo1,p);
	strategyGen(strategyClass,strategyNo2,q);

	// effect of error and infromation channels are included
	effectiveStrategyGen(errorChannels,p);
	effectiveStrategyGen(errorChannels,q);

	// transition matrix generation
	MatrixXd M(8,8);
	for (int i = 0; i < 8; ++i)
	{
		int initStage = 1;
		if (i>3)
		{
			initStage = 2;
		}
		double focalCinit = 0; // focal player cooperated initially
		if (i == 0 or i == 1 or i == 4 or i == 5)
		{
			focalCinit = 1;
		}
		for (int j = 0; j < 8; ++j)
		{
			int finalStage = 1;
			if (j>3)
			{
				finalStage = 2;
			}

			// calculation of the first factor of the element of the transition matrix
			double x  = 1;
			if (finalStage == 1){x = x*qvec(i);}
			if (finalStage == 2){x = x*(1-qvec(i));}

			// calculation of the second factor of the element of the transition matrix
			int iF = i;
			if (initStage >  finalStage){iF = iF-4;}
			if (initStage <  finalStage){iF = iF+4;}
			double yF = 1-p(iF);
			double focalCfinal = 0; // focal player cooperated finally
			if (j == 0 or j == 1 or j == 4 or j == 5){
				yF = p(iF); focalCfinal = 1;
			}

			// calculation of the second factor of the element of the transition matrix
			int iO = i;
			if (i == 1){iO = 2;}
			if (i == 2){iO = 1;}
			if (i == 5){iO = 6;}
			if (i == 6){iO = 5;}
			if (initStage >  finalStage){iO = iO-4;}
			if (initStage <  finalStage){iO = iO+4;}
			if (scheme == 2)
			{	
				// scheme of alternating moves is applied
				if (focalCinit>focalCfinal){iO = iO+1;}
				if (focalCinit<focalCfinal){iO = iO-1;}
			}
			double yO = 1-q(iO);
			if (j == 0 or j == 2 or j == 4 or j == 6){
				yO = q(iO);
			}

			M(i,j) = x*yF*yO;
		}
		
	}
	// cout<<M<<endl;
	// Transposition of transition matrix
	MatrixXd MT(8,8);
	MT=M.transpose();

	// right eigen vectors of transposition of transition matrix calculation with eiegen-value unity
	EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
	MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
	int evc;
	double maxz = 0.0;
	for (int i = 0; i < 8; ++i)
	{
		
		// cout<<es.eigenvectors().col(i).real().transpose()<<endl;
		if (z(i) > maxz)
		{
			maxz = z(i);
			evc=i;
		}
	}
	VectorXd V = es.eigenvectors().col(evc).real();
	double SUM = V.sum();V=V/SUM;
	double payoff1,payoff2,coop1,coop2,stage1freq;
	payCoopb1(V,payoff1,payoff2,coop1,coop2,stage1freq,b1);
	P12C12S1<<payoff1,payoff2,coop1,coop2,stage1freq;
}
////////////////////////////////////////////////////////////////////////



// genrates outputs for whole (b1-n2) cjannels space
// input: (scheme of moves), (strategy class), (q vector), (error rate), (selection stregth)
// saves equilibrium outcomes: coorperation rate, efficacy(first channel), efficay second channel
////////////////////////////////////////////////////////////////////////
void b1n2(int scheme,int strategyClass,int qvecNo,double eps,double beta)
{
	// parmeters that are less frequently changed and some initializations
	double dx = 1.0/double(150); // elementary grid size in the channels space
	int N = 100;
	double w = 0.0;
	double n1 = 0.1;
	int classSize;
	classSizeGen(strategyClass,classSize);
	ofstream file1;
	file1.open("cooperation_b1_n2.txt");
	for (double b1 = 1.2; b1 < 1.2+1.0+dx; b1 += dx)
	{
		for (double n2 = 0; n2 < 0.9+dx; n2 += dx)
		{
			double c = 0;
			VectorXd P12C12S1(5);
			VectorXd errorChannels(6);
			MatrixXd PayM(classSize,classSize);
			MatrixXd RhoM(classSize,classSize);
			MatrixXd CoopM(classSize,classSize);
			MatrixXd S1freqM(classSize,classSize);
			errorChannels<<eps,eps,n1,n2,w,w;
			// PayM CoopM, S1freqM are calcculated for all the startegies
			/////////////////////////////////////////////
			for (int i = 0; i < classSize; ++i)
			{
				for (int j = i; j < classSize; ++j)
				{

					payCoopGenb1(strategyClass,errorChannels,i,j,qvecNo,scheme,P12C12S1,b1);

					PayM(i,j)    = P12C12S1(0);PayM(j,i)    = P12C12S1(1);
					CoopM(i,j)   = P12C12S1(2);CoopM(j,i)   = P12C12S1(3);
					S1freqM(i,j) = P12C12S1(4);S1freqM(j,i) = P12C12S1(4);
				}
			}
			// Calculation of Rho matrix
			/////////////////////////////////////////////
			for (int i = 0; i < classSize; ++i)
			{
				for (int j = 0; j < classSize; ++j)
				{
					if (i!=j)
					{
						RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
					}
				}
				RhoM(i,i) = 0;
				double summ = 0;
				for (int j = 0; j < classSize; ++j)
				{
					summ = summ +RhoM(i,j);
				}
				RhoM(i,i)=1.0-summ;
			}
			MatrixXd MT(classSize,classSize);
			MT=RhoM.transpose();
			EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
			MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
			int evc;
			double maxz = 0.0;
			for (int i = 0; i < classSize; ++i)
			{
				if (z(i) > maxz)
				{
					maxz = z(i);
					evc=i;
				}
			}
			VectorXd A = es.eigenvectors().col(evc).real();
			double SUM = A.sum();
			A = A/SUM;// relative abundances of all the strategies are computed
			for (int i = 0; i < classSize; ++i)
			{
				c += A(i)*CoopM(i,i);
			}
			file1<<c<<" ";
			cout<<"=================================="<<endl;
			cout<<"b1 = "<<b1<<" cooeparation = "<<c<<endl;
			cout<<"=================================="<<endl;
		}
		file1<<endl;
	}
	file1.close();
}
////////////////////////////////////////////////////////////////////////



// genrates outputs for whole (eps-n2) cjannels space
// input: (scheme of moves), (strategy class), (q vector), (error rate), (selection stregth)
// saves equilibrium outcomes: coorperation rate, efficacy(first channel), efficay second channel
////////////////////////////////////////////////////////////////////////
void epsn2(int scheme,int strategyClass,int qvecNo,double beta)
{
	// parmeters that are less frequently changed and some initializations
	double dx = 1.0/double(100); // elementary grid size in the channels space
	double dxP = 5/double(100);
	int N = 100;
	double w = 0.0;
	double n1 = 0.1;
	int classSize;
	double b1 = 2.0;
	classSizeGen(strategyClass,classSize);
	ofstream file1;
	file1.open("cooperation_eps_n2.txt");
	for (double epsP = -4; epsP < -1; epsP += dxP)
	{
		double eps = pow(10,epsP);
		for (double n2 = 0; n2 < 0.9+dx; n2 += dx)
		{
			double c = 0;
			VectorXd P12C12S1(5);
			VectorXd errorChannels(6);
			MatrixXd PayM(classSize,classSize);
			MatrixXd RhoM(classSize,classSize);
			MatrixXd CoopM(classSize,classSize);
			MatrixXd S1freqM(classSize,classSize);
			errorChannels<<eps,eps,n1,n2,w,w;
			// PayM CoopM, S1freqM are calcculated for all the startegies
			/////////////////////////////////////////////
			for (int i = 0; i < classSize; ++i)
			{
				for (int j = i; j < classSize; ++j)
				{

					payCoopGenb1(strategyClass,errorChannels,i,j,qvecNo,scheme,P12C12S1,b1);

					PayM(i,j)    = P12C12S1(0);PayM(j,i)    = P12C12S1(1);
					CoopM(i,j)   = P12C12S1(2);CoopM(j,i)   = P12C12S1(3);
					S1freqM(i,j) = P12C12S1(4);S1freqM(j,i) = P12C12S1(4);
				}
			}
			// Calculation of Rho matrix
			/////////////////////////////////////////////
			for (int i = 0; i < classSize; ++i)
			{
				for (int j = 0; j < classSize; ++j)
				{
					if (i!=j)
					{
						RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
					}
				}
				RhoM(i,i) = 0;
				double summ = 0;
				for (int j = 0; j < classSize; ++j)
				{
					summ = summ +RhoM(i,j);
				}
				RhoM(i,i)=1.0-summ;
			}
			MatrixXd MT(classSize,classSize);
			MT=RhoM.transpose();
			EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
			MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
			int evc;
			double maxz = 0.0;
			for (int i = 0; i < classSize; ++i)
			{
				if (z(i) > maxz)
				{
					maxz = z(i);
					evc=i;
				}
			}
			VectorXd A = es.eigenvectors().col(evc).real();
			double SUM = A.sum();
			A = A/SUM;// relative abundances of all the strategies are computed
			for (int i = 0; i < classSize; ++i)
			{
				c += A(i)*CoopM(i,i);
			}
			file1<<c<<" ";
			cout<<"=================================="<<endl;
			cout<<"eps = "<<eps<<" cooeparation = "<<c<<endl;
			cout<<"=================================="<<endl;
		}
		file1<<endl;
	}
	file1.close();
}
////////////////////////////////////////////////////////////////////////



// Discounted payoff and cooperation genration given two strategy number
// input: (strategy class), (error and information channels), (focal strategy no.), (opponent strategy no.), (q vector no.), moves scheme \in {1: simultaneous, 2: alternating}
// output: (expected payoff1, payoff2, cooperation1, cooperation2)
////////////////////////////////////////////////////////////////////////
void payCoopGenDis(int strategyClass,VectorXd errorChannels,int strategyNo1,int strategyNo2,int qvecNo,int scheme,double delta,VectorXd &P12C12S1)
{
	// the transition vector is generated
	VectorXd qvec(8);
	qvecGen(qvecNo,qvec);
	// pure strategy pair is generated
	VectorXd p(8),q(8); 
	strategyGen(strategyClass,strategyNo1,p);
	strategyGen(strategyClass,strategyNo2,q);
	// effect of error and infromation channels are included
	effectiveStrategyGen(errorChannels,p);
	effectiveStrategyGen(errorChannels,q);
	VectorXd V0(8);
	double p0 = p(0), q0 = q(0);
	if (scheme == 1)
	{
		V0(0) = p(0)*q(0);
		V0(1) = p(0)*(1-q(0));
		V0(2) = (1-p(0))*q(0);
		V0(3) = (1-p(0))*(1-q(0));
	}
	if (scheme == 2)
	{
		V0(0) = p(0)*q(0);
		V0(1) = p(0)*(1-q(0));
		V0(2) = (1-p(0))*q(1);
		V0(3) = (1-p(0))*(1-q(1));
	}
	V0(4) = 0;
	V0(5) = 0;
	V0(6) = 0;
	V0(7) = 0;
	// transition matrix generation
	MatrixXd M(8,8);
	for (int i = 0; i < 8; ++i)
	{
		int initStage = 1;
		if (i>3)
		{
			initStage = 2;
		}
		double focalCinit = 0; // focal player cooperated initially
		if (i == 0 or i == 1 or i == 4 or i == 5)
		{
			focalCinit = 1;
		}
		for (int j = 0; j < 8; ++j)
		{
			int finalStage = 1;
			if (j>3)
			{
				finalStage = 2;
			}

			// calculation of the first factor of the element of the transition matrix
			double x  = 1;
			if (finalStage == 1){x = x*qvec(i);}
			if (finalStage == 2){x = x*(1-qvec(i));}

			// calculation of the second factor of the element of the transition matrix
			int iF = i;
			if (initStage >  finalStage){iF = iF-4;}
			if (initStage <  finalStage){iF = iF+4;}
			double yF = 1-p(iF);
			double focalCfinal = 0; // focal player cooperated finally
			if (j == 0 or j == 1 or j == 4 or j == 5){
				yF = p(iF); focalCfinal = 1;
			}

			// calculation of the second factor of the element of the transition matrix
			int iO = i;
			if (i == 1){iO = 2;}
			if (i == 2){iO = 1;}
			if (i == 5){iO = 6;}
			if (i == 6){iO = 5;}
			if (initStage >  finalStage){iO = iO-4;}
			if (initStage <  finalStage){iO = iO+4;}
			if (scheme == 2)
			{	
				// scheme of alternating moves is applied
				if (focalCinit>focalCfinal){iO = iO+1;}
				if (focalCinit<focalCfinal){iO = iO-1;}
			}
			double yO = 1-q(iO);
			if (j == 0 or j == 2 or j == 4 or j == 6){
				yO = q(iO);
			}

			M(i,j) = x*yF*yO;
		}
		
	}
	// cout<<M<<endl;
	// Transposition of transition matrix
	MatrixXd I = MatrixXd::Identity(8,8);
	VectorXd V = ((1.0-delta)*V0.transpose()*(I-delta*M).inverse()).transpose();
	double SUM = V.sum();V=V/SUM;
	double payoff1,payoff2,coop1,coop2,stage1freq;

	if (delta==1)
	{
		MatrixXd MT(8,8);
		MT=M.transpose();
		EigenSolver<MatrixXd> es(MT);
		MatrixXd z=es.eigenvalues().real();
		int evc;
		double maxz = 0.0;
		for (int i = 0; i < 8; ++i)
		{
			if (z(i) > maxz)
			{
				maxz = z(i);
				evc=i;
			}
		}
		V = es.eigenvectors().col(evc).real();
		SUM = V.sum();V=V/SUM;
	}
	payCoop(V,payoff1,payoff2,coop1,coop2,stage1freq);
	P12C12S1<<payoff1,payoff2,coop1,coop2,stage1freq;
}
////////////////////////////////////////////////////////////////////////



// Discount genrates outputs for whole (n-delta) cjannels space
// input: (scheme of moves), (strategy class), (q vector), (error rate), (selection stregth)
// saves equilibrium outcomes: coorperation rate, efficacy(first channel), efficay second channel
////////////////////////////////////////////////////////////////////////
void DissymmetricErrorChannelsSpace(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1,const std::string& filename2,const std::string& filename3)
{
	// parmeters that are less frequently changed and some initializations
	double dx = 0.5/double(100); // elementary grid size in the channels space
	double dxP = 1.0/double(50);
	int N = 100;
	double w = 0.0;
	int classSize;
	classSizeGen(strategyClass,classSize);
	ofstream file1;
	file1.open(filename1);
	ofstream file2;
	file2.open(filename2);
	ofstream file3;
	file3.open(filename3);
	for (double delta = 0.0; delta < 1.0+dxP; delta += dxP)
	{
		for (double n = 0; n < 0.5+dx; n += dx)
		{
			double c = 0;
			double alphaIn = 0;
			double mInformationN = 0;
			double capacityN = 1.0;
			VectorXd P12C12S1_FI(5);
			VectorXd P12C12S1_FJ(5);
			VectorXd errorChannels(6);
			MatrixXd PayM(classSize,classSize);
			MatrixXd RhoM(classSize,classSize);
			MatrixXd CoopM(classSize,classSize);
			MatrixXd S1freqM(classSize,classSize);
			errorChannels<<eps,eps,n,n,w,w;
			// PayM CoopM, S1freqM are calcculated for all the startegies
			/////////////////////////////////////////////
			for (int i = 0; i < classSize; ++i)
			{
				for (int j = i; j < classSize; ++j)
				{
					payCoopGenDis(strategyClass,errorChannels,i,j,qvecNo,scheme,delta,P12C12S1_FI);
					payCoopGenDis(strategyClass,errorChannels,j,i,qvecNo,scheme,delta,P12C12S1_FJ);

					PayM(i,j)    = (P12C12S1_FI(0)+P12C12S1_FJ(1))/2.0;PayM(j,i)    = (P12C12S1_FI(1)+P12C12S1_FJ(0))/2.0;
					CoopM(i,j)   = (P12C12S1_FI(2)+P12C12S1_FJ(3))/2.0;CoopM(j,i)   = (P12C12S1_FI(3)+P12C12S1_FJ(2))/2.0;
					S1freqM(i,j) = (P12C12S1_FI(4)+P12C12S1_FJ(4))/2.0;S1freqM(j,i) = (P12C12S1_FI(4)+P12C12S1_FJ(4))/2.0;
				}
			}
			// Calculation of Rho matrix
			/////////////////////////////////////////////
			for (int i = 0; i < classSize; ++i)
			{
				for (int j = 0; j < classSize; ++j)
				{
					if (i!=j)
					{
						RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
					}
				}
				RhoM(i,i) = 0;
				double summ = 0;
				for (int j = 0; j < classSize; ++j)
				{
					summ = summ +RhoM(i,j);
				}
				RhoM(i,i)=1.0-summ;
			}
			MatrixXd MT(classSize,classSize);
			MT=RhoM.transpose();
			EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
			MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
			int evc;
			double maxz = 0.0;
			for (int i = 0; i < classSize; ++i)
			{
				if (z(i) > maxz)
				{
					maxz = z(i);
					evc=i;
				}
			}
			VectorXd A = es.eigenvectors().col(evc).real();
			double SUM = A.sum();
			A = A/SUM;// relative abundances of all the strategies are computed
			for (int i = 0; i < classSize; ++i)
			{
				c += A(i)*CoopM(i,i);
				alphaIn = S1freqM(i,i);
				mInformationN += A(i)*Mutual(alphaIn,n,n);
			}
			capacityN = Capacity(n,n);
			file1<<c<<" ";
			file2<<mInformationN<<" ";
			file3<<capacityN<<" ";
			cout<<"=================================="<<endl;
			cout<< "n=" << n << endl;
			cout<< "delta=" << delta <<endl;
			cout<< "c=" << c <<endl;
			cout<<"=================================="<<endl;
		}
		file1<<endl;
		file2<<endl;
		file3<<endl;
	}
	file1.close();
	file2.close();
	file3.close();
}
////////////////////////////////////////////////////////////////////////



// Discount genrates outputs for whole (n-delta) cjannels space
// input: (scheme of moves), (strategy class), (q vector), (error rate), (selection stregth)
// saves equilibrium outcomes: coorperation rate, efficacy(first channel), efficay second channel
////////////////////////////////////////////////////////////////////////
void DissymmetricErrorChannelsSpaceW(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1,const std::string& filename2,const std::string& filename3)
{
	// parmeters that are less frequently changed and some initializations
	double dx = 0.5/double(100); // elementary grid size in the channels space
	double dxP = 1.0/double(50);
	int N = 100;
	double n = 0.0;
	int classSize;
	classSizeGen(strategyClass,classSize);
	ofstream file1;
	file1.open(filename1);
	ofstream file2;
	file2.open(filename2);
	ofstream file3;
	file3.open(filename3);
	for (double delta = 0.0; delta < 1.0+dxP; delta += dxP)
	{
		for (double w = 0; w < 0.5+dx; w += dx)
		{
			double c = 0;
			double alphaIn = 0;
			double mInformationW = 0;
			double capacityW = 1.0;
			VectorXd P12C12S1_FI(5);
			VectorXd P12C12S1_FJ(5);
			VectorXd errorChannels(6);
			MatrixXd PayM(classSize,classSize);
			MatrixXd RhoM(classSize,classSize);
			MatrixXd CoopM(classSize,classSize);
			MatrixXd S1freqM(classSize,classSize);
			errorChannels<<eps,eps,n,n,w,w;
			// PayM CoopM, S1freqM are calcculated for all the startegies
			/////////////////////////////////////////////
			for (int i = 0; i < classSize; ++i)
			{
				for (int j = i; j < classSize; ++j)
				{
					payCoopGenDis(strategyClass,errorChannels,i,j,qvecNo,scheme,delta,P12C12S1_FI);
					payCoopGenDis(strategyClass,errorChannels,j,i,qvecNo,scheme,delta,P12C12S1_FJ);

					PayM(i,j)    = (P12C12S1_FI(0)+P12C12S1_FJ(1))/2.0;PayM(j,i)    = (P12C12S1_FI(1)+P12C12S1_FJ(0))/2.0;
					CoopM(i,j)   = (P12C12S1_FI(2)+P12C12S1_FJ(3))/2.0;CoopM(j,i)   = (P12C12S1_FI(3)+P12C12S1_FJ(2))/2.0;
					S1freqM(i,j) = (P12C12S1_FI(4)+P12C12S1_FJ(4))/2.0;S1freqM(j,i) = (P12C12S1_FI(4)+P12C12S1_FJ(4))/2.0;
				}
			}
			// Calculation of Rho matrix
			/////////////////////////////////////////////
			for (int i = 0; i < classSize; ++i)
			{
				for (int j = 0; j < classSize; ++j)
				{
					if (i!=j)
					{
						RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
					}
				}
				RhoM(i,i) = 0;
				double summ = 0;
				for (int j = 0; j < classSize; ++j)
				{
					summ = summ +RhoM(i,j);
				}
				RhoM(i,i)=1.0-summ;
			}
			MatrixXd MT(classSize,classSize);
			MT=RhoM.transpose();
			EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
			MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
			int evc;
			double maxz = 0.0;
			for (int i = 0; i < classSize; ++i)
			{
				if (z(i) > maxz)
				{
					maxz = z(i);
					evc=i;
				}
			}
			VectorXd A = es.eigenvectors().col(evc).real();
			double SUM = A.sum();
			A = A/SUM;// relative abundances of all the strategies are computed
			for (int i = 0; i < classSize; ++i)
			{
				c += A(i)*CoopM(i,i);
				alphaIn = CoopM(i,i);
				mInformationW += A(i)*Mutual(alphaIn,w,w);
			}
			capacityW = Capacity(n,n);
			file1<<c<<" ";
			file2<<mInformationW<<" ";
			file3<<capacityW<<" ";
			cout<<"=================================="<<endl;
			cout<< "w=" << w << endl;
			cout<< "delta=" << delta <<endl;
			cout<< "c=" << c <<endl;
			cout<<"=================================="<<endl;
		}
		file1<<endl;
		file2<<endl;
		file3<<endl;
	}
	file1.close();
	file2.close();
	file3.close();
}
////////////////////////////////////////////////////////////////////////



// genrates outputs for whole (n1-n2) cjannels space
// input: (scheme of moves), (strategy class), (q vector), (error rate), (selection stregth)
// saves equilibrium outcomes: coorperation rate, efficacy(first channel), efficay second channel
////////////////////////////////////////////////////////////////////////
void AsymmetricErrorChannelsSpaceNW(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1,const std::string& filename2,const std::string& filename3,const std::string& filename4,const std::string& filename5,const std::string& filename6,const std::string& filename7)
{
	// parmeters that are less frequently changed and some initializations
	double dx = 0.5/double(100); // elementary grid size in the channels space
	int N = 100;
	int classSize;
	classSizeGen(strategyClass,classSize);
	ofstream file1;
	file1.open(filename1);//"cooperation.txt"
	ofstream file2;
	file2.open(filename2);//"mInformationN.txt"
	ofstream file3;
	file3.open(filename3);//"mInformationW.txt"
	ofstream file4;
	file4.open(filename4);//"capacityN.txt"
	ofstream file5;
	file5.open(filename5);//"capacityW.txt"
	ofstream file6;
	file6.open(filename6);//"abundant.txt"
	ofstream file7;
	file7.open(filename7);//"percent.txt"
	for (double n = 0; n < 0.5+dx; n += dx)
	{
		for (double w = 0; w < 0.5+dx; w += dx)
		{
			double c = 0;
			double alphaInN = 0;
			double alphaInW = 0;
			double mInformationN = 0;
			double capacityN = 1.0;
			double mInformationW = 0;
			double capacityW = 1.0;
			int abundant1 = 15;
			double percent1 = 0;

			VectorXd P12C12S1(5);
			VectorXd errorChannels(6);
			MatrixXd PayM(classSize,classSize);
			MatrixXd RhoM(classSize,classSize);
			MatrixXd CoopM(classSize,classSize);
			MatrixXd S1freqM(classSize,classSize);
			errorChannels<<eps,eps,n,n,w,w;

			// PayM CoopM, S1freqM are calcculated for all the startegies
			/////////////////////////////////////////////
			for (int i = 0; i < classSize; ++i)
			{
				for (int j = i; j < classSize; ++j)
				{

					payCoopGen(strategyClass,errorChannels,i,j,qvecNo,scheme,P12C12S1);

					PayM(i,j)    = P12C12S1(0);PayM(j,i)    = P12C12S1(1);
					CoopM(i,j)   = P12C12S1(2);CoopM(j,i)   = P12C12S1(3);
					S1freqM(i,j) = P12C12S1(4);S1freqM(j,i) = P12C12S1(4);
				}
			}
			// Calculation of Rho matrix
			/////////////////////////////////////////////
			for (int i = 0; i < classSize; ++i)
			{
				for (int j = 0; j < classSize; ++j)
				{
					if (i!=j)
					{
						RhoM(i,j) = calcRho(PayM,j,i,N,beta)/double(classSize);
					}
				}
				RhoM(i,i) = 0;
				double summ = 0;
				for (int j = 0; j < classSize; ++j)
				{
					summ = summ +RhoM(i,j);
				}
				RhoM(i,i)=1.0-summ;
			}
			MatrixXd MT(classSize,classSize);
			MT=RhoM.transpose();
			EigenSolver<MatrixXd> es(MT);// cout<<es.eigenvectors().real()<<endl;
			MatrixXd z=es.eigenvalues().real();// cout<<es.eigenvalues().real()<<endl;
			int evc;
			double maxz = 0.0;
			for (int i = 0; i < classSize; ++i)
			{
				if (z(i) > maxz)
				{
					maxz = z(i);
					evc=i;
				}
			}
			VectorXd A = es.eigenvectors().col(evc).real();
			double SUM = A.sum();
			A = A/SUM;// relative abundances of all the strategies are computed

			for (int i = 0; i < classSize; ++i)
			{
				c += A(i)*CoopM(i,i);
				alphaInN = S1freqM(i,i);
				alphaInW = CoopM(i,i);
				mInformationN += A(i)*Mutual(alphaInN,n,n);
				mInformationW += A(i)*Mutual(alphaInW,w,w);
				if (A(i)>=A(abundant1))
				{
					abundant1 = i;
				}
			}
			capacityN = Capacity(n,n);
			capacityW = Capacity(w,w);
			percent1 = A(abundant1);
			file1<<c<<" ";
			file2<<mInformationN<<" ";
			file3<<mInformationW<<" ";
			file4<<capacityN<<" ";
			file5<<capacityW<<" ";
			file6<<abundant1<<" ";
			file7<<percent1<<" ";
			cout<<"=================================="<<endl;
			cout<<"n ="<< n << " " << "w =" << w <<endl;
			cout<<"=================================="<<endl;
		}
		file1<<endl;
		file2<<endl;
		file3<<endl;
		file4<<endl;
		file5<<endl;
		file6<<endl;
		file7<<endl;
	}
	file1.close();
	file2.close();
	file3.close();
	file4.close();
	file5.close();
	file6.close();
	file7.close();
}
////////////////////////////////////////////////////////////////////////



// Parameters changing in a single place for the channels space module
// input: (scheme of moves)
////////////////////////////////////////////////////////////////////////
void AspaceDataParamsSaveNW(int scheme)
{
	double beta = 10;             // selection stregth \in {0.1,1.0,10.0}
	double eps  = pow(10,-3);     // error in execution of any action \in {0.1,0.01,0.001}
	
	//AsymmetricErrorChannelsSpace(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1,const std::string& filename2,const std::string& filename3,const std::string& filename4,,const std::string& filename5)
	AsymmetricErrorChannelsSpaceNW(scheme,2,1,eps,beta,"cooperation112.txt","mInformationN112.txt","mInformationW112.txt","capacityN112.txt","capacityW112.txt","abundant112.txt","percent112.txt");
	AsymmetricErrorChannelsSpaceNW(scheme,2,2,eps,beta,"cooperation122.txt","mInformationN122.txt","mInformationW122.txt","capacityN122.txt","capacityW122.txt","abundant122.txt","percent122.txt");
	AsymmetricErrorChannelsSpaceNW(scheme,2,3,eps,beta,"cooperation132.txt","mInformationN132.txt","mInformationW132.txt","capacityN132.txt","capacityW132.txt","abundant132.txt","percent132.txt");
}
////////////////////////////////////////////////////////////////////////



// Discount Parameters changing in a single place for the channels space module
// input: (scheme of moves)
////////////////////////////////////////////////////////////////////////
void disspaceDataParamsSaveW(int scheme)
{
	double beta = 10;             // selection stregth \in {0.1,1.0,10.0}
	double eps  = pow(10,-3);    // error in execution of any action \in {0.1,0.01,0.001}
	
	// DissymmetricErrorChannelsSpace(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1,const std::string& filename2,const std::string& filename3)
	DissymmetricErrorChannelsSpaceW(scheme,2,1,eps,beta,"cooperation112.txt","mInformationW112.txt","capacityW112.txt");
	DissymmetricErrorChannelsSpaceW(scheme,2,2,eps,beta,"cooperation122.txt","mInformationW122.txt","capacityW122.txt");
	DissymmetricErrorChannelsSpaceW(scheme,2,3,eps,beta,"cooperation132.txt","mInformationW132.txt","capacityW132.txt");

}
////////////////////////////////////////////////////////////////////////



// Discount Parameters changing in a single place for the channels space module
// input: (scheme of moves)
////////////////////////////////////////////////////////////////////////
void disspaceDataParamsSave(int scheme)
{
	double beta = 10;             // selection stregth \in {0.1,1.0,10.0}
	double eps  = pow(10,-3);    // error in execution of any action \in {0.1,0.01,0.001}
	
	// DissymmetricErrorChannelsSpace(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1,const std::string& filename2,const std::string& filename3)
	DissymmetricErrorChannelsSpace(scheme,2,1,eps,beta,"cooperation112.txt","mInformationN112.txt","capacityN112.txt");
	DissymmetricErrorChannelsSpace(scheme,2,2,eps,beta,"cooperation122.txt","mInformationN122.txt","capacityN122.txt");
	DissymmetricErrorChannelsSpace(scheme,2,3,eps,beta,"cooperation132.txt","mInformationN132.txt","capacityN132.txt");

}
////////////////////////////////////////////////////////////////////////



// Parameters changing in a single place for the b1n2 and epsn2 space modules
// input: (scheme of moves)
////////////////////////////////////////////////////////////////////////
void mechanism(int scheme)
{
	double beta = 10;             // selection stregth \in {0.1,1.0,10.0}
	double eps  = pow(10,-3);     // error in execution of any action \in {0.1,0.01,0.001}
	// b1n2(int scheme,int strategyClass,int qvecNo,double eps,double beta)
	b1n2(scheme,2,1,eps,beta);
	// epsn2(int scheme,int strategyClass,int qvecNo,double beta)
	epsn2(scheme,2,1,beta);
}
////////////////////////////////////////////////////////////////////////



// Parameters changing in a single place for the tmatabundance module
// input: (scheme of moves)
////////////////////////////////////////////////////////////////////////
void MarkovChain(int scheme)
{
	double beta = 10;             // selection stregth \in {0.1,1.0,10.0}
	double eps  = pow(10,-3);     // error in execution of any action \in {0.1,0.01,0.001}

	//tMatAbundanceSave(int scheme,int strategyClass,int qvecNo,double eps,double beta,double n1,double n2,const std::string& filename1,const std::string& filename2)
	tMatAbundanceSave(scheme,2,1,eps,beta,0.1,0.2,"transitionMatrix_0.1_0.2.txt","abundanceVector_0.1_0.2.txt");
	tMatAbundanceSave(scheme,2,1,eps,beta,0.1,0.4,"transitionMatrix_0.1_0.4.txt","abundanceVector_0.1_0.4.txt");
	tMatAbundanceSave(scheme,2,1,eps,beta,0.1,0.6,"transitionMatrix_0.1_0.6.txt","abundanceVector_0.1_0.6.txt");
	tMatAbundanceSave(scheme,2,1,eps,beta,0.1,0.8,"transitionMatrix_0.1_0.8.txt","abundanceVector_0.1_0.8.txt");
}
////////////////////////////////////////////////////////////////////////



// Parameters changing in a single place for the channels space module
// input: (scheme of moves)
////////////////////////////////////////////////////////////////////////
void AspaceDataParamsSave(int scheme)
{
	double beta = 10;             // selection stregth \in {0.1,1.0,10.0}
	double eps  = pow(10,-3);     // error in execution of any action \in {0.1,0.01,0.001}
	
	//AsymmetricErrorChannelsSpace(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1,const std::string& filename2,const std::string& filename3,const std::string& filename4,,const std::string& filename5)
	AsymmetricErrorChannelsSpace(scheme,2,1,eps,beta,"cooperation112.txt","mInformationN112.txt","capacityN112.txt","abundant112.txt","percent112.txt");
	AsymmetricErrorChannelsSpace(scheme,2,2,eps,beta,"cooperation122.txt","mInformationN122.txt","capacityN122.txt","abundant122.txt","percent122.txt");
	AsymmetricErrorChannelsSpace(scheme,2,3,eps,beta,"cooperation132.txt","mInformationN132.txt","capacityN132.txt","abundant132.txt","percent132.txt");
}
////////////////////////////////////////////////////////////////////////



// Parameters changing in a single place for the channels space module
// input: (scheme of moves)
////////////////////////////////////////////////////////////////////////
void AspaceDataParamsSaveW(int scheme)
{
	double beta = 10;             // selection stregth \in {0.1,1.0,10.0}
	double eps  = pow(10,-3);     // error in execution of any action \in {0.1,0.01,0.001}
	
	//AsymmetricErrorChannelsSpace(int scheme,int strategyClass,int qvecNo,double eps,double beta,const std::string& filename1,const std::string& filename2,const std::string& filename3,const std::string& filename4,,const std::string& filename5)
	AsymmetricErrorChannelsSpaceW(scheme,2,1,eps,beta,"cooperation112.txt","mInformationW112.txt","capacityW112.txt","abundant112.txt","percent112.txt");
	AsymmetricErrorChannelsSpaceW(scheme,2,2,eps,beta,"cooperation122.txt","mInformationW122.txt","capacityW122.txt","abundant122.txt","percent122.txt");
	AsymmetricErrorChannelsSpaceW(scheme,2,3,eps,beta,"cooperation132.txt","mInformationW132.txt","capacityW132.txt","abundant132.txt","percent132.txt");
}
////////////////////////////////////////////////////////////////////////



// Parameters changing in a single place for the channels space module
// input: (scheme of moves)
////////////////////////////////////////////////////////////////////////
void spaceDataParamsSave(int scheme)
{
	double beta = 10;             // selection stregth \in {0.1,1.0,10.0}
	double eps  = pow(10,-3);    // error in execution of any action \in {0.1,0.01,0.001}
	
	// symmetricErrorChannelsSpace(int scheme,int strategyClass,int qvecNo,double eps,double beta, const std::string& filename1,std::string& filename2)
	symmetricErrorChannelsSpace(scheme,2,1,eps,beta,"cooperation112.txt","alphaIn112.txt");
	symmetricErrorChannelsSpace(scheme,2,2,eps,beta,"cooperation122.txt","alphaIn122.txt");
	symmetricErrorChannelsSpace(scheme,2,3,eps,beta,"cooperation132.txt","alphaIn132.txt");

	symmetricErrorChannelsSpace(scheme,4,1,eps,beta,"cooperation114.txt","alphaIn114.txt");
	symmetricErrorChannelsSpace(scheme,4,2,eps,beta,"cooperation124.txt","alphaIn124.txt");
	symmetricErrorChannelsSpace(scheme,4,3,eps,beta,"cooperation134.txt","alphaIn134.txt");
}
////////////////////////////////////////////////////////////////////////


// Parameters changing in a single place for the channels space module
// input: (scheme of moves)
////////////////////////////////////////////////////////////////////////
void spaceDataParamsSaveCT(int scheme)
{
	double beta = 10;             // selection stregth \in {0.1,1.0,10.0}
	double eps  = pow(10,-3);    // error in execution of any action \in {0.1,0.01,0.001}
	
	// symmetricErrorChannelsSpace(int scheme,int strategyClass,int qvecNo,double eps,double beta, const std::string& filename1,std::string& filename2)
	symmetricErrorChannelsSpaceCT(scheme,2,1,eps,beta,"cooperation112.txt","alphaIn112.txt");
	symmetricErrorChannelsSpaceCT(scheme,2,2,eps,beta,"cooperation122.txt","alphaIn122.txt");
	symmetricErrorChannelsSpaceCT(scheme,2,3,eps,beta,"cooperation132.txt","alphaIn132.txt");

	symmetricErrorChannelsSpaceCT(scheme,4,1,eps,beta,"cooperation114.txt","alphaIn114.txt");
	symmetricErrorChannelsSpaceCT(scheme,4,2,eps,beta,"cooperation124.txt","alphaIn124.txt");
	symmetricErrorChannelsSpaceCT(scheme,4,3,eps,beta,"cooperation134.txt","alphaIn134.txt");
}
////////////////////////////////////////////////////////////////////////



// Saves trajectory data
// input: (scheme of moves)
void trajDataParamsSave(int scheme)
{
	double beta = 10;             // selection stregth
	double eps  = pow(10,-3);     // error in execution of any action
	
	// symmetricErrorChannelsTrajGen(scheme, strategyClass, qvecNo, eps, n, w, beta, "filename.txt");
	symmetricErrorChannelsTrajGen(scheme,2, 1, eps, 0.5, 0, beta, "withNoise112.txt");
	symmetricErrorChannelsTrajGen(scheme,2, 2, eps, 0.5, 0, beta, "withNoise122.txt");
	symmetricErrorChannelsTrajGen(scheme,2, 3, eps, 0.5, 0, beta, "withNoise132.txt");

	symmetricErrorChannelsTrajGen(scheme,4, 1, eps, 0.5, 0, beta, "withNoise114.txt");
	symmetricErrorChannelsTrajGen(scheme,4, 2, eps, 0.5, 0, beta, "withNoise124.txt");
	symmetricErrorChannelsTrajGen(scheme,4, 3, eps, 0.5, 0, beta, "withNoise134.txt");

	symmetricErrorChannelsTrajGen(scheme,2, 1, eps, 0.0, 0, beta, "withOutNoise112.txt");
	symmetricErrorChannelsTrajGen(scheme,2, 2, eps, 0.0, 0, beta, "withOutNoise122.txt");
	symmetricErrorChannelsTrajGen(scheme,2, 3, eps, 0.0, 0, beta, "withOutNoise132.txt");

	symmetricErrorChannelsTrajGen(scheme,4, 1, eps, 0.0, 0, beta, "withOutNoise114.txt");
	symmetricErrorChannelsTrajGen(scheme,4, 2, eps, 0.0, 0, beta, "withOutNoise124.txt");
	symmetricErrorChannelsTrajGen(scheme,4, 3, eps, 0.0, 0, beta, "withOutNoise134.txt");
}
////////////////////////////////////////////////////////////////////////



// Saves trajectory data
// input: (scheme of moves)
void trajDataParamsSaveW(int scheme)
{
	double beta = 10;             // selection stregth
	double eps  = pow(10,-3);     // error in execution of any action
	
	// symmetricErrorChannelsTrajGen(scheme, strategyClass, qvecNo, eps, n, w, beta, "filename.txt");
	symmetricErrorChannelsTrajGen(scheme,2, 1, eps, 0, 0.5, beta, "withNoise112.txt");
	symmetricErrorChannelsTrajGen(scheme,2, 2, eps, 0, 0.5, beta, "withNoise122.txt");
	symmetricErrorChannelsTrajGen(scheme,2, 3, eps, 0, 0.5, beta, "withNoise132.txt");

	symmetricErrorChannelsTrajGen(scheme,4, 1, eps, 0, 0.5, beta, "withNoise114.txt");
	symmetricErrorChannelsTrajGen(scheme,4, 2, eps, 0, 0.5, beta, "withNoise124.txt");
	symmetricErrorChannelsTrajGen(scheme,4, 3, eps, 0, 0.5, beta, "withNoise134.txt");

	symmetricErrorChannelsTrajGen(scheme,2, 1, eps, 0, 0.0, beta, "withOutNoise112.txt");
	symmetricErrorChannelsTrajGen(scheme,2, 2, eps, 0, 0.0, beta, "withOutNoise122.txt");
	symmetricErrorChannelsTrajGen(scheme,2, 3, eps, 0, 0.0, beta, "withOutNoise132.txt");

	symmetricErrorChannelsTrajGen(scheme,4, 1, eps, 0, 0.0, beta, "withOutNoise114.txt");
	symmetricErrorChannelsTrajGen(scheme,4, 2, eps, 0, 0.0, beta, "withOutNoise124.txt");
	symmetricErrorChannelsTrajGen(scheme,4, 3, eps, 0, 0.0, beta, "withOutNoise134.txt");
}
////////////////////////////////////////////////////////////////////////



// Parameters changing in a single place for the channels space module
// input: (scheme of moves)
////////////////////////////////////////////////////////////////////////
void spaceDataParamsSaveW(int scheme)
{
	double beta = 10;             // selection stregth \in {0.1,1.0,10.0}
	double eps  = pow(10,-3);    // error in execution of any action \in {0.1,0.01,0.001}
	
	// symmetricErrorChannelsSpace(int scheme,int strategyClass,int qvecNo,double eps,double beta, const std::string& filename1,std::string& filename2)
	symmetricErrorChannelsSpaceW(scheme,2,1,eps,beta,"cooperation112.txt","alphaIn112.txt");
	symmetricErrorChannelsSpaceW(scheme,2,2,eps,beta,"cooperation122.txt","alphaIn122.txt");
	symmetricErrorChannelsSpaceW(scheme,2,3,eps,beta,"cooperation132.txt","alphaIn132.txt");

	symmetricErrorChannelsSpaceW(scheme,4,1,eps,beta,"cooperation114.txt","alphaIn114.txt");
	symmetricErrorChannelsSpaceW(scheme,4,2,eps,beta,"cooperation124.txt","alphaIn124.txt");
	symmetricErrorChannelsSpaceW(scheme,4,3,eps,beta,"cooperation134.txt","alphaIn134.txt");
}
////////////////////////////////////////////////////////////////////////