#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

// use ANN to check whether there is connection between linearly related values
// Only two layers -- input layer and output layer -- are considered
#define Number 100000
// here I found that the sigmoid function is not better than linear function if the output is actually linearly dependent on input

int main()
{
	int i,j,n;
	float eta=1.0;
	float w[3][3];
	float x[Number][3],y[Number][3];
	float A[Number][3],Opt[Number][3],sum;
	float Esum1,Esum2,deltaw,deltav;


	srand(time(NULL));

	for(n=0;n<Number;n++)  // randomly create input and output data
	{
		for(j=0;j<3;j++)
		{
			x[n][j] = (float) (rand()/(RAND_MAX+0.0));
//			y[n][j] = (float) (rand()/(RAND_MAX+0.0));
//			y[n][j] = x[n][j]/2.0;     // y depends linearly on x
			y[n][j] = x[n][j] * x[n][j];  // non-linear dependence

		}
//		printf(" %f %f %f : %f %f %f\n",x[n][0],x[n][1],x[n][2],y[n][0],y[n][1],y[n][2]);
	}

	for(i=0;i<3;i++)
		for(j=0;j<3;j++)
			w[i][j] = (float) (rand()/(RAND_MAX+0.0));

	for(n=0;n<Number;n++)
	{
		for(j=0;j<3;j++)  // the initial values in the neurons of the output layer
		{
			A[n][j]=Opt[n][j]=0.0;
			for(i=0;i<3;i++)
				A[n][j] += x[n][i] * w[i][j];
			Opt[n][j] = 1.0 / (1.0 + exp(-1.0 * A[n][j]));    // use sigmoid function
		//	Opt[n][j] = A[n][j]; // use linear function
		}
	
	}

	Esum1=0.0;
	for(n=0;n<Number;n++)
		for(j=0;j<3;j++)   // the initial error
			Esum1 += (Opt[n][j]-y[n][j])*(Opt[n][j]-y[n][j]); 
	Esum2=Esum1+1.0;

	printf("Learning the data...\n");	
	while(fabs(Esum1-Esum2) > 0.01)  // then start to change the weights according to the backpropagation algorithm
	{
		Esum2=Esum1;

		for(i=0;i<3;i++)  // update the w[][] weights
			for(j=0;j<3;j++)
			{
				deltaw=0.0;
				for(n=0;n<Number;n++)
					deltaw += -2.0 * eta * (Opt[n][j]-y[n][j]) * Opt[n][j] * (1-Opt[n][j]) * x[n][i];   // derivative of sigmoid function
				//	deltaw += -2.0 * eta * (Opt[n][j]-y[n][j]) * x[n][i];  // use linear activation function
				w[i][j] = w[i][j] + deltaw/Number;
			}

		for(n=0;n<Number;n++)
		{
			for(j=0;j<3;j++)  // update the values in the neurons of the output layer
			{
				A[n][j]=Opt[n][j]=0.0;
				for(i=0;i<3;i++)
					A[n][j] += x[n][i] * w[i][j];
				Opt[n][j] = 1.0 / (1.0 + exp(-1.0 * A[n][j]));    // use sigmoid function
			//	Opt[n][j] = A[n][j];   // use linear function
			}

		}

		Esum1=0.0;
		for(n=0;n<Number;n++)
			for(j=0;j<3;j++)   // update the error
				Esum1 += (Opt[n][j]-y[n][j])*(Opt[n][j]-y[n][j]); 

		printf("   %f   %f   %f  \n", Esum1,Esum2,fabs(Esum1-Esum2));
	
	}

	float ierror;
	int n1,n2,n3;
	n1=n2=n3=0;
	for(n=0;n<Number;n++)
	{
		ierror = 0.0;
		for(j=0;j<3;j++)
			ierror += (Opt[n][j]-y[n][j])*(Opt[n][j]-y[n][j]);
		if(ierror < 0.01)
			n1++;
		if(ierror < 0.0001)
			n2++;
		if(ierror < 0.000001)
			n3++;
			
	}
	printf(" 0.01 %d : 0.0001 %d : 0.000001 %d : %d\n",n1,n2,n3,Number);


	printf("\nfinal weights w[][]\n");
	for(i=0;i<3;i++)
	{
		for(j=0;j<3;j++)
			printf(" %f ", w[i][j]);
		printf("\n");
	}
	printf("\n");

	float xi[3],yi[3],Ai[3],Opti[3];
	int iter=1;

	while(iter<3)
	{
	printf("Now enter an input value:\n");
	scanf("%f %f %f",&xi[0],&xi[1],&xi[2]);
	printf("calculating\n");
	for(j=0;j<3;j++)
	{
		Ai[j]=Opti[j]=0.0;
		for(i=0;i<3;i++)
			Ai[j] += xi[i] * w[i][j];
		Opti[j] = 1.0 / (1.0 + exp(-1.0 * Ai[j]));  // sigmoid function
		yi[j] = Opti[j];
	}
	printf(" Input xi: %f %f %f\n Guess yi: %f %f %f\n\n",xi[0],xi[1],xi[2],yi[0],yi[1],yi[2]);

	iter++;
	}

	return(0);

}


