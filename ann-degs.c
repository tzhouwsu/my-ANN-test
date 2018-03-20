#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>

#define Nsnaps 10000
#define Inputdur 100 // the input duration of 100 snapshots
#define Outputdur 10 // the output duration of 1 snapshots

#define Nhidd 20 // number of neurons in the hidden layer

#define TOL 0.01 // convergence tolerance


int myread(int xdata[Nsnaps-Inputdur][Inputdur],int ydata[Nsnaps-Inputdur][Outputdur])
{
	FILE *fp;
	int degs[Nsnaps];
	int mi,mj,mk,mdeg,mnum,merror;
	char buffer[1000];

	if((fp=fopen("degs","r"))==NULL)
	{
		printf("Cannot find the degrees file\n");
		return(-1);
	}

	for(mi=0;mi<Nsnaps;mi++)
		degs[mi] = -1;

	rewind(fp);
	mnum=0;merror=0;
	while(fscanf(fp,"%d",&mdeg) == 1)
	{
		fgets(buffer,sizeof(buffer),fp);
		if(mnum>=Nsnaps)
		{
			merror = 1;
			break;
		}
		degs[mnum] = mdeg;
		mnum += 1;
	}

	if(merror==1 || mnum!=Nsnaps)
	{
		printf("Error reading the degrees file\n");
		fclose(fp);
		return(-2);
	}
	else
	{
		for(mi=0;mi<Nsnaps-Inputdur;mi++)
		{
			for(mj=0;mj<Inputdur;mj++)
				xdata[mi][mj] = degs[mi+mj];
			for(mk=0;mk<Outputdur;mk++)
				ydata[mi][mk] = degs[mi+Inputdur+mk];
		}
		fclose(fp);
		return(0);
	}

}

int mytrain(int xdata[Nsnaps-Inputdur][Inputdur], int ydata[Nsnaps-Inputdur][Outputdur], int Ntrain, int Ntest, float wij[Inputdur][Nhidd], float vjk[Nhidd][Outputdur])
{
	int ti,tj,tk,tn;
	float eta = 0.001;
	float c1[Inputdur][Nhidd],c2[Nhidd][Outputdur];
	float A[Ntrain][Nhidd],Mid[Ntrain][Nhidd],B[Ntrain][Outputdur],Opt[Ntrain][Outputdur];
	float sum1,sum2,avg1,avg2,dguess;
	int n1,n2,n3;

	srand(time(NULL));	
	for(ti=0;ti<Inputdur;ti++)     // initialize the coefficient
		for(tj=0;tj<Nhidd;tj++)
			c1[ti][tj] = 0.1;
//			c1[ti][tj] = (float) (rand()/(RAND_MAX+0.0));
	for(tj=0;tj<Nhidd;tj++)
		for(tk=0;tk<Outputdur;tk++)
			c2[tj][tk] = 0.1;
//			c2[tj][tk] = (float) (rand()/(RAND_MAX+0.0));

	for(tn=0;tn<Ntrain;tn++)  // initialize the hidden layer and the output layer
	{

		for(tj=0;tj<Nhidd;tj++)
		{
			A[tn][tj]=Mid[tn][tj]=0.0;
			for(ti=0;ti<Inputdur;ti++)
				A[tn][tj] += xdata[tn][ti] * c1[ti][tj];
			A[tn][tj] = A[tn][tj] / Inputdur;
			Mid[tn][tj] = 1.0 / (1.0 + exp(-1.0 * A[tn][tj]));  // use sigmoid function in the hidden layer
		}

		for(tk=0;tk<Outputdur;tk++)
		{
			B[tn][tk]=Opt[tn][tk]=0.0;
			for(tj=0;tj<Nhidd;tj++)
				B[tn][tk] += Mid[tn][tj] * c2[tj][tk];
			B[tn][tk] = B[tn][tk] / Nhidd;
			Opt[tn][tk] = 1.0 / (1.0 + exp(-1.0 * B[tn][tk]));  // sigmoid function
		}
	}

	sum1=0.0;
	for(tn=0;tn<Ntrain;tn++)
		for(tk=0;tk<Outputdur;tk++)
			sum1 += (100*Opt[tn][tk]-ydata[tn][tk])*(100*Opt[tn][tk]-ydata[tn][tk])/Ntrain;

	sum2=sum1+1.0;
	while(fabs(sum2-sum1) > TOL)
	{
	printf("...learning the data... %f \n", fabs(sum2-sum1));
//		printf("  %f %f %f\n",sum1,sum2,fabs(sum1-sum2));
		sum2=sum1;
		for(ti=0;ti<Inputdur;ti++)   // update c1[][]
			for(tj=0;tj<Nhidd;tj++)
			{
				avg1=0.0;
				for(tn=0;tn<Ntrain;tn++)
					for(tk=0;tk<Outputdur;tk++)
						avg1 += -2.0 * eta * (100.0*Opt[tn][tk]-ydata[tn][tk]) * Opt[tn][tk] * (1-Opt[tn][tk]) * c2[tj][tk] * Mid[tn][tj] * (1-Mid[tn][tj]) * xdata[tn][ti];
				c1[ti][tj] += avg1/Ntrain;
			}
		for(tj=0;tj<Nhidd;tj++)  // update c2[][]
			for(tk=0;tk<Outputdur;tk++)
			{
				avg2=0.0;
				for(tn=0;tn<Ntrain;tn++)
					avg2 += -2.0 * eta * (100.0*Opt[tn][tk]-ydata[tn][tk]) * Opt[tn][tk] * (1-Opt[tn][tk]) * Mid[tn][tj];
				c2[tj][tk] += avg2/Ntrain;
			}

		for(tn=0;tn<Ntrain;tn++)  // update the hidden layer and the output layer
		{
			for(tj=0;tj<Nhidd;tj++)
			{
				A[tn][tj]=Mid[tn][tj]=0.0;
				for(ti=0;ti<Inputdur;ti++)
					A[tn][tj] += xdata[tn][ti] * c1[ti][tj];
				Mid[tn][tj] = 1.0 / (1.0 + exp(-1.0 * A[tn][tj]));  // use sigmoid function in the hidden layer
			}
			for(tk=0;tk<Outputdur;tk++)
			{
				B[tn][tk]=Opt[tn][tk]=0.0;
				for(tj=0;tj<Nhidd;tj++)
					B[tn][tk] += Mid[tn][tj] * c2[tj][tk];
				Opt[tn][tk] = 1.0 / (1.0 + exp(-1.0 * B[tn][tk]));  // sigmoid function multifplied by 100, the guessed output is an integer
			}
		}
		sum1=0.0;
		for(tn=0;tn<Ntrain;tn++)   //update the error
			for(tk=0;tk<Outputdur;tk++)
				sum1 += (100.0*Opt[tn][tk]-ydata[tn][tk])*(100.0*Opt[tn][tk]-ydata[tn][tk])/Ntrain;

//		printf("  %f %f %f\n",sum1,sum2,fabs(sum1-sum2));
	}

	for(ti=0;ti<Inputdur;ti++)
		for(tj=0;tj<Nhidd;tj++)
			wij[ti][tj] = c1[ti][tj];
	for(tj=0;tj<Nhidd;tj++)
		for(tk=0;tk<Outputdur;tk++)
			vjk[tj][tk] = c2[tj][tk];

	n1=n2=n3=0;
	for(tn=0;tn<Ntrain;tn++)
	{
		dguess=0.0;
		for(tk=0;tk<Outputdur;tk++)
			dguess += fabs(100.0*Opt[tn][tk]-ydata[tn][tk]);
		dguess = dguess/Outputdur;  // average deviate from the data
		if(dguess < 0.5) // guess integer number correctly 
			n1 += 1;
		if(dguess < 1.5) // guess +- 1
			n2 += 1;
		if(dguess < 2.5)
			n3 += 1; // guess +- 2
	}

	printf(" Training result: %d %d %d : %d\n",n1,n2,n3,Ntrain);
	return(0);
}

int mytest(int xdata[Nsnaps-Inputdur][Inputdur], int ydata[Nsnaps-Inputdur][Outputdur], int Ntrain, int Ntest, float wij[Inputdur][Nhidd], float vjk[Nhidd][Outputdur])
{
	int ti,tj,tk,tn;
	float A[Ntest][Nhidd],Mid[Ntest][Nhidd],B[Ntest][Outputdur],Opt[Ntest][Outputdur];
	float sum1,sum2,avg1,avg2,dguess;
	int n1,n2,n3;


	for(tn=0;tn<Ntest;tn++)  // calculate the hidden layer and the output layer
	{
		for(tj=0;tj<Nhidd;tj++)
		{
			A[tn][tj]=Mid[tn][tj]=0.0;
			for(ti=0;ti<Inputdur;ti++)
				A[tn][tj] += xdata[tn+Ntrain][ti] * wij[ti][tj];
			Mid[tn][tj] = 1.0 / (1.0 + exp(-1.0 * A[tn][tj]));  // use sigmoid function in the hidden layer
		}
		for(tk=0;tk<Outputdur;tk++)
		{
			B[tn][tk]=Opt[tn][tk]=0.0;
			for(tj=0;tj<Nhidd;tj++)
				B[tn][tk] += Mid[tn][tj] * vjk[tj][tk];
			Opt[tn][tk] = 1.0 / (1.0 + exp(-1.0 * B[tn][tk]));  // sigmoid function multifplied by 100, the guessed output is an integer
		}
	}

	n1=n2=n3=0;
	for(tn=0;tn<Ntest;tn++)
	{
		dguess=0.0;
		for(tk=0;tk<Outputdur;tk++)
			dguess += fabs(100.0*Opt[tn][tk]-ydata[tn+Ntrain][tk]);
		dguess = dguess/Outputdur;  // average deviate from the data
		if(dguess < 0.5) // guess integer number correctly 
			n1 += 1;
		if(dguess < 1.5) // guess +- 1
			n2 += 1;
		if(dguess < 2.5)
			n3 += 1; // guess +- 2
	}

	printf(" Testing result: %d %d %d : %d\n",n1,n2,n3,Ntest);
	return(0);
}


int main()
{
	int x[Nsnaps-Inputdur][Inputdur],y[Nsnaps-Inputdur][Outputdur];
	int i,Trainset,Testset;
	float w[Inputdur][Nhidd],v[Nhidd][Outputdur];  // the trained coefficient

	printf("\nMain: reading the data\n");
	if(myread(x,y) != 0)  // read the data
	{
		printf("---reading error---\n");
		return(-1);
	}

	Trainset = 5000;  //500 data for traning
	Testset = 4000; //400 data for testing

	printf("\nMain: learning the training data\n");
	if(mytrain(x,y,Trainset,Testset,w,v) != 0)
	{
		printf("---training error---\n");
		return(-2);
	}

	printf("\nMain: evaluating the test data\n");
	if(mytest(x,y,Trainset,Testset,w,v) != 0)
	{
		printf("---testing error---\n");
		return(-3);
	}



	return(0);
}


