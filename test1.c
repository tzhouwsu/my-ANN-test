#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

// use backpropagation algorithm for fingding the weights in ANN

int main()
{
	int i,j,k;
	float eta=1.0;
	float w[3][2],v[2][3];
	float x[3],y[3];
	x[0]=1.0;x[1]=0.25;x[2]=-0.5;  // input
	y[0]=1.0;y[1]=-1.0;y[2]=0.0;   // output, let it range from -2.0 to 2.0

	printf("\nInput values:\n");
	for(i=0;i<3;i++)
		printf(" %f ",x[i]);
	printf("\n\n");

	printf("\nOutput values:\n");
	for(k=0;k<3;k++)
		printf(" %f ",y[k]);
	printf("\n\n");

	srand(time(NULL));

	for(i=0;i<3;i++)
		for(j=0;j<2;j++)
			w[i][j] = (float) (rand()/(RAND_MAX+1.0));
	for(j=0;j<2;j++)
		for(k=0;k<3;k++)
			v[j][k] = (float) (rand()/(RAND_MAX+1.0));

	float A[2],Mid[2],B[3],Opt[3],Fin[3];
	float E1,E2,deltaw,deltav;

	for(j=0;j<2;j++)  // the initial values in the neurons of the middle layer
	{
		A[j]=Mid[j]=0.0;
		for(i=0;i<3;i++)
			A[j] += x[i] * w[i][j];
		Mid[j] = 1.0 / (1.0 + exp(-1.0 * A[j]));    // use sigmoid function
	}

	for(k=0;k<3;k++)     // the initial values in the neurons of the output layer
	{
		B[k]=Opt[k]=Fin[k]=0.0;
		for(j=0;j<2;j++)
			B[k] += Mid[j] * v[j][k];
		Fin[k] = 1.0 / (1.0 + exp(-1.0 * B[k]));
		Opt[k] = (Fin[k] - 0.5) * 4.0;  // use sigmoid function, re-scale it to (-2.0, 2.0)
	}

	E1=0.0;
	for(k=0;k<3;k++)   // the initial error
		E1 += (Opt[k]-y[k])*(Opt[k]-y[k]); 
	E2=E1+1.0;
	
	while(fabs(E1-E2) > 0.00001)  // then start to change the weights according to the backpropagation algorithm
	{
		E2=E1;

		for(i=0;i<3;i++)  // update the w[][] weights
			for(j=0;j<2;j++)
			{
				deltaw = 0.0;
				for(k=0;k<3;k++)
					deltaw += -2.0 * eta * (Opt[k]-y[k]) * 4.0 * Fin[k] * (1-Fin[k]) * v[j][k] * Mid[j] * (1-Mid[j]) * x[i];  // a factor of 4 from rescale the Opt value
				w[i][j] = w[i][j] + deltaw;
			}

		for(j=0;j<2;j++)    // update the v[][] weights
			for(k=0;k<3;k++)
			{
				deltav = -2.0 * eta * (Opt[k]-y[k]) * 4.0 * Fin[k] * (1-Fin[k]) * Mid[j];   // a factor of 4 from rescale the Opt value
				v[j][k] = v[j][k] + deltav;
			}

		
		for(j=0;j<2;j++)  // update the values in the neurons of the middle layer
		{
			A[j]=Mid[j]=0.0;
			for(i=0;i<3;i++)
				A[j] += x[i] * w[i][j];
			Mid[j] = 1.0 / (1.0 + exp(-1.0 * A[j]));    // use sigmoid function
		}

		for(k=0;k<3;k++)     // update the values in the neurons of the output layer
		{
			B[k]=Opt[k]=Fin[k]=0.0;
			for(j=0;j<2;j++)
				B[k] += Mid[j] * v[j][k];
			Fin[k] = 1.0 / (1.0 + exp(-1.0 * B[k]));
			Opt[k] = (Fin[k] - 0.5) * 4.0;  // use sigmoid function, re-scale it to (-2.0, 2.0)
		}

		E1=0.0;
		for(k=0;k<3;k++)   // update the error
			E1 += (Opt[k]-y[k])*(Opt[k]-y[k]); 

		printf("   %f   %f   %f  \n", E1,E2,fabs(E1-E2));

	}

	printf("\nfinal ANN output values:\n");
	for(k=0;k<3;k++)
		printf(" %f ",Opt[k]);
	printf("\n\n");

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


