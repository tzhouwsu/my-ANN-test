#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

// use ANN to check whether there is connection between linearly related values
// Use three layers -- input layer, hidden layer, output layer -- are considered
#define Number 10000   // training set
#define Test 2000   // test set

int main()
{
	int i,j,k,n;
	float eta=10.0;
	float w[3][3],v[3][3];
	float x[Number+Test][3],y[Number+Test][3];
	float A[Number+Test][3],Mid[Number+Test][3],B[Number+Test][3],Opt[Number+Test][3],sum;
	float Esum1,Esum2,deltaw,deltav;


	srand(time(NULL));

	for(n=0;n<Number+Test;n++)  // randomly create input and output data
	{
		for(j=0;j<3;j++)
		{
			x[n][j] = (float) (rand()/(RAND_MAX+0.0));
//			y[n][j] = (float) (rand()/(RAND_MAX+0.0));
			y[n][j] = x[n][j]/2.0 + 0.2;     // y depends linearly on x
//			y[n][j] = x[n][j] * (1.0 - x[n][j]) ;  // non-linear dependence

		}
//		printf(" %f %f %f : %f %f %f\n",x[n][0],x[n][1],x[n][2],y[n][0],y[n][1],y[n][2]);
	}

	for(i=0;i<3;i++)
		for(j=0;j<3;j++)
			w[i][j] = (float) (rand()/(RAND_MAX+0.0));
	for(j=0;j<3;j++)
		for(k=0;k<3;k++)
			v[j][k] = (float) (rand()/(RAND_MAX+0.0));


	for(n=0;n<Number+Test;n++)
	{
		for(j=0;j<3;j++)  // the initial values in the neurons of the hidden layer
		{
			A[n][j]=Mid[n][j]=0.0;
			for(i=0;i<3;i++)
				A[n][j] += x[n][i] * w[i][j];
			Mid[n][j] = 1.0 / (1.0 + exp(-1.0 * A[n][j]));    // use sigmoid function
		//	Mid[n][j] = A[n][j]; // use linear function
		}
	
		for(k=0;k<3;k++)  // the initial values in the neurons of the output layer
		{
			B[n][k]=Opt[n][k]=0.0;
			for(j=0;j<3;j++)
				B[n][k] += Mid[n][j] * v[j][k];
			Opt[n][k] = 1.0 / (1.0 + exp(-1.0 * B[n][k]));    // use sigmoid function
		//	Opt[n][k] = B[n][k]; // use linear function
		}
	
	}

	Esum1=0.0;
	for(n=0;n<Number;n++)     // error is calculated within the training set
		for(k=0;k<3;k++)   // the initial error
			Esum1 += (Opt[n][k]-y[n][k])*(Opt[n][k]-y[n][k])/Number; 
	Esum2=Esum1+1.0;

	printf("Learning the data...\n");	
	while(fabs(Esum1-Esum2) > 0.01)  // then start to change the weights according to the backpropagation algorithm
	{
//		eta = fabs(Esum1-Esum2) * 100.0;
		Esum2=Esum1;
	
		for(i=0;i<3;i++)  // update the w[][] weights
			for(j=0;j<3;j++)
			{
				deltaw=0.0;
				for(n=0;n<Number;n++)
					for(k=0;k<3;k++)
						deltaw += -2.0 * eta * (Opt[n][k]-y[n][k]) * Opt[n][k] * (1-Opt[n][k]) * v[j][k] * Mid[n][j] * (1-Mid[n][j]) * x[n][i];   // derivative of sigmoid function
					//	deltaw += -2.0 * eta * (Opt[n][k]-y[n][k]) * v[j][k] * x[n][i];  // use linear activation function
				w[i][j] = w[i][j] + deltaw/Number;
			}

		for(j=0;j<3;j++)  // update the v[][] weights
			for(k=0;k<3;k++)
			{
				deltav=0.0;
				for(n=0;n<Number;n++)
					deltav += -2.0 * eta * (Opt[n][k]-y[n][k]) * Opt[n][k] * (1-Opt[n][k]) * Mid[n][j];   // derivative of sigmoid function
				//	deltav += -2.0 * eta * (Opt[n][k]-y[n][k]) * Mid[n][j];  // use linear activation function
				v[j][k] = v[j][k] + deltav/Number;
			}

		for(n=0;n<Number+Test;n++)
		{
			for(j=0;j<3;j++)  // update the values in the neurons of the hidden layer
			{
				A[n][j]=Mid[n][j]=0.0;
				for(i=0;i<3;i++)
					A[n][j] += x[n][i] * w[i][j];
				Mid[n][j] = 1.0 / (1.0 + exp(-1.0 * A[n][j]));    // use sigmoid function
			//	Mid[n][j] = A[n][j];   // use linear function
			}
		}

		for(n=0;n<Number+Test;n++)
		{
			for(k=0;k<3;k++)  // update the values in the neurons of the output layer
			{
				B[n][k]=Opt[n][k]=0.0;
				for(j=0;j<3;j++)
					B[n][k] += Mid[n][j] * v[j][k];
				Opt[n][k] = 1.0 / (1.0 + exp(-1.0 * B[n][k]));    // use sigmoid function
			//	Opt[n][k] = B[n][k];   // use linear function
			}
		}


		Esum1=0.0;
		for(n=0;n<Number;n++)  // error is calculated within the training set
			for(k=0;k<3;k++)   // update the error
				Esum1 += (Opt[n][k]-y[n][k])*(Opt[n][k]-y[n][k])/Number; 

		printf("   %f   %f   %f  \n", Esum1,Esum2,fabs(Esum1-Esum2));
	
	}

	float ierror;
	int n1,n2,n3;
	n1=n2=n3=0;
	for(n=Number;n<Number+Test;n++)
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
	printf("\n\n 0.01 %d : 0.0001 %d : 0.000001 %d : %d\n",n1,n2,n3,Test);

	printf("\nfinal weights w[][]\n");
	for(i=0;i<3;i++)
	{
		for(j=0;j<3;j++)
			printf(" %f ", w[i][j]);
		printf("\n");
	}
	printf("\n");

	printf("\nfinal weights v[][]\n");
	for(i=0;i<3;i++)
	{
		for(j=0;j<3;j++)
			printf(" %f ", v[i][j]);
		printf("\n");
	}
	printf("\n");


	float xi[3],yi[3],Ai[3],Midi[3],Bi[3],Opti[3];
	int iter=1;

	while(iter<5)
	{
		printf("Now enter an input value:\n");
		scanf("%f %f %f",&xi[0],&xi[1],&xi[2]);
		printf("calculating\n");
		for(j=0;j<3;j++)
		{
			Ai[j]=Midi[j]=0.0;
			for(i=0;i<3;i++)
				Ai[j] += xi[i] * w[i][j];
			Midi[j] = 1.0 / (1.0 + exp(-1.0 * Ai[j]));  // sigmoid function
			printf("  %f %f\n",Ai[j],Midi[j]);
		}
		for(k=0;k<3;k++)
		{
			Bi[k]=Opti[k]=0.0;
			for(j=0;j<3;j++)
				Bi[k] += Midi[j] * v[j][k];
			Opti[k] = 1.0 / (1.0 + exp(-1.0 * Bi[k]));  // sigmoid function
			yi[k] = Opti[k];
		}
		printf(" Input xi: %f %f %f\n Guess yi: %f %f %f\n\n",xi[0],xi[1],xi[2],yi[0],yi[1],yi[2]);

		iter++;
	}

	return(0);

}


