#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

//�ú�������Ϊ��__global__����ʾ��GPU����ִ��.
//�书��Ϊ������pA��pB�ж�Ӧλ�õ�������ӣ����������������pC�Ķ�Ӧλ����
//ÿ�������������СΪsize
__global__
void add(const float * pA, const float * pB, float * pC, unsigned int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;		//���㵱ǰ�����е�����
	if (index < size)										//ȷ����һ����Ч������
		pC[index] = pA[index] + pB[index];

}

int main()
{
	unsigned int numElement = 30000000;
	int totalSize = sizeof(float)* numElement;

	//init
	float *pA = (float*)malloc(totalSize);
	float *pB = (float*)malloc(totalSize);
	float *pC = (float*)malloc(totalSize);

	for (int i = 0; i < numElement; ++i)
	{
		*(pA + i) = rand() / (float)RAND_MAX;;
		*(pB + i) = rand() / (float)RAND_MAX;
	}

	//cpu segment

	//begin use cpu comput
	clock_t startTime, endTime;
	startTime = clock();
	for (int i = 0; i < numElement; ++i)
	{
		*(pC + i) = *(pA + i) + *(pB + i);
	}
	endTime = clock();
	//end use cpu comput

	printf("use cpu comput finish!\n");
	printf("use total time = %fs\n", (endTime - startTime) / 1000.f);
	printf("\n\n");


	//gpu segment
	float *pD, *pE, *pF;
	cudaError_t err = cudaSuccess;

	//malloc memory
	err = cudaMalloc(&pD, totalSize);
	if (err != cudaSuccess)
	{
		printf("call cudaMalloc fail for pD.\n");
		exit(1);
	}

	err = cudaMalloc(&pE, totalSize);
	if (err != cudaSuccess)
	{
		printf("call cudaMalloc fail for pE.\n");
		exit(1);
	}

	err = cudaMalloc(&pF, totalSize);
	if (err != cudaSuccess)
	{
		printf("call cudaMalloc fail for pF.\n");
		exit(1);
	}

	//copy data  from pA pB pC to pD pE pF
	err = cudaMemcpy(pD, pA, totalSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("call cudaMemcpy fail for pA to pD.\n");
		exit(1);
	}

	err = cudaMemcpy(pE, pB, totalSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("call cudaMemcpy fail for pB to pE.\n");
		exit(1);
	}


	//begin use gpu comput
	startTime = clock();
	int threadPerBlock = 1024;
	int numBlock = (numElement - 1) / threadPerBlock + 1;
	add << <numBlock, threadPerBlock >> >(pD, pE, pF, numElement);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("use gpu comput fail!\n");
		exit(1); 
	}

	endTime = clock();
	printf("use gpu comput finish!\n");
	printf("use time : %fs\n",(endTime - startTime) / 1000.f);
	//end use gpu comput


	//copu data from device to host
	err = cudaMemcpy(pC, pF, numElement, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		printf("call cudaMemcpy form pF to pC fail.\n");
		exit(1);
	}

	//check data
	for (int i = 0; i < numElement; ++i)
	{
		if (fabs(pA[i] + pB[i] - pC[i]) > 1e-5)
		{
			printf("%f + %f != %f\n",pA[i],pB[i],pC[i]);
		}
	}

	//�ͷ��豸�ϵ��ڴ�
	cudaFree(pD);
	cudaFree(pE);
	cudaFree(pF);

	//�ڳ����˳�ǰ�����øú������ø��豸��ʹ����ȥ�����豸״̬�������ڳ����˳�ǰ���е����ݽ���ˢ����
	err = cudaDeviceReset();
	if (err != cudaSuccess)
	{
		printf("call cudaDeviceReset fail!\n");
		exit(1);
	}

	free(pA);
	free(pB);
	free(pC);

	getchar();
	return 0;
}