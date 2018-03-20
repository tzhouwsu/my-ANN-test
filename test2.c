#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

// use ANN to check whether there is connection between two sets of random numbers
#define Number 100000
#define Nhidd 2

int main()
{
	int i,j,k,n;
	float eta=1.0;
	float w[3][2],v[3][2];
	float x[Number][3],y[Number][3];
	float A[Number][Nhidd],Mid[Number][Nhidd],B[Number][3],Opt[Number][3];
	float Esum1,Esum2,deltaw,deltav;


	srand(time(NULL));

	for(n=0;n<Number;n++)  // randomly create input and output data
	{
		for(j=0;j<3;j++)
		{
			x[n][j] = (float) (rand()/(RAND_MAX+0.0));
//			y[n][j] = (float) (rand()/(RAND_MAX+0.0));
			y[n][j] = x[n][j]/2.0;
		}

		printf(" %f %f %f : %f %f %f\n",x[n][0],x[n][1],x[n][2],y[n][0],y[n][1],y[n][2]);
	}

	for(i=0;i<3;i++)
		for(j=0;j<Nhidd;j++)
			w[i][j] = (float) (rand()/(RAND_MAX+0.0));
	for(j=0;j<Nhidd;j++)
		for(k=0;k<3;k++)
			v[j][k] = (float) (rand()/(RAND_MAX+0.0));

	for(n=0;n<Number;n++)
	{
		for(j=0;j<Nhidd;j++)  // the initial values in the neurons of the middle layer
		{
			A[n][j]=Mid[n][j]=0.0;
			for(i=0;i<3;i++)
				A[n][j] += x[n][i] * w[i][j];
			Mid[n][j] = 1.0 / (1.0 + exp(-1.0 * A[n][j]));    // use sigmoid function
		}
	
		for(k=0;k<3;k++)     // the initial values in the neurons of the output layer
		{
			B[n][k]=Opt[n][k]=0.0;
			for(j=0;j<Nhidd;j++)
				B[n][k] += Mid[n][j] * v[j][k];
			Opt[n][k] = 1.0 / (1.0 + exp(-1.0 * B[n][k]));  // use sigmoid function
		}
	}

	Esum1=0.0;
	for(n=0;n<Number;n++)
		for(k=0;k<3;k++)   // the initial error
			Esum1 += (Opt[n][k]-y[n][k])*(Opt[n][k]-y[n][k]); 
	Esum2=Esum1+1.0;
	
	while(fabs(Esum1-Esum2) > 0.00001)  // then start to change the weights according to the backpropagation algorithm
	{
		Esum2=Esum1;

		for(i=0;i<3;i++)  // update the w[][] weights
			for(j=0;j<Nhidd;j++)
			{
				deltaw=0.0;
				for(n=0;n<Number;n++)
					for(k=0;k<3;k++)
						deltaw += -2.0 * eta * (Opt[n][k]-y[n][k]) * Opt[n][k] * (1-Opt[n][k]) * v[j][k] * Mid[n][j] * (1-Mid[n][j]) * x[n][i];
				w[i][j] = w[i][j] + deltaw/Number;
			}

		for(j=0;j<Nhidd;j++)    // update the v[][] weights
			for(k=0;k<3;k++)
			{
				deltav=0.0;
				for(n=0;n<Number;n++)
					deltav += -2.0 * eta * (Opt[n][k]-y[n][k]) * Opt[n][k] * (1-Opt[n][k]) * Mid[n][j];
				v[j][k] = v[j][k] + deltav/Number;
			}

		for(n=0;n<Number;n++)
		{
			for(j=0;j<Nhidd;j++)  // update the values in the neurons of the middle layer
			{
				A[n][j]=Mid[n][j]=0.0;
				for(i=0;i<3;i++)
					A[n][j] += x[n][i] * w[i][j];
				Mid[n][j] = 1.0 / (1.0 + exp(-1.0 * A[n][j]));    // use sigmoid function
			}

			for(k=0;k<3;k++)     // update the values in the neurons of the output layer
			{
				B[n][k]=Opt[n][k]=0.0;
				for(j=0;j<Nhidd;j++)
					B[n][k] += Mid[n][j] * v[j][k];
				Opt[n][k] = 1.0 / (1.0 + exp(-1.0 * B[n][k]));  // use sigmoid function
			}
		}

		Esum1=0.0;
		for(n=0;n<Number;n++)
			for(k=0;k<3;k++)   // update the error
				Esum1 += (Opt[n][k]-y[n][k])*(Opt[n][k]-y[n][k]); 

		printf("   %f   %f   %f  \n", Esum1,Esum2,fabs(Esum1-Esum2));
	
	}


	printf("\nfinal weights w[][]\n");
	for(i=0;i<3;i++)
	{
		for(j=0;j<2;j++)
			printf(" %f ", w[i][j]);
		printf("\n");
	}
	printf("\n");

	printf("\nfinal weights v[][]\n");
	for(j=0;j<2;j++)
	{
		for(k=0;k<3;k++)
			printf(" %f ", v[j][k]);
		printf("\n");
	}
	printf("\n");

	return(0);

}


