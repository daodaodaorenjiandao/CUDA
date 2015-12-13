#include <iostream>
#include <cuda_runtime.h>

using namespace std;


int main()
{
	float * pDeviceData = nullptr;
	int width = 10 * sizeof(float);
	int height = 10 * sizeof(float);
	size_t pitch;

	cudaError err = cudaSuccess;

	//1 use cudaMallocPitch function
	err = cudaMallocPitch(&pDeviceData, &pitch, width, height);		//注意这里的width和height的单位为字节数
	if (err != cudaSuccess)
	{
		cout << "call cudaMallocPitch fail!!!" << endl;
		exit(1);
	}
	cout << "width: " << width << endl;
	cout << "height: " << height << endl;
	cout << "pitch: " << pitch << endl;

	

	//2 use cudaMalloc3D
	cudaPitchedPtr pitchPtr;
	cudaExtent extent;
	extent.width = 10 * sizeof(float);
	extent.height = 22 * sizeof(float);
	extent.depth = 33 * sizeof(float);

	err = cudaMalloc3D(&pitchPtr, extent);
	if (err != cudaSuccess)
	{
		cout << "call cudaMalloc3D fail!!!" << endl;
		exit(1);
	}
	cout << "\n\n";
	cout << "width: " << extent.width << endl;			//输出申请内存的初始值
	cout << "height: " << extent.height << endl;
	cout << "depth: " << extent.depth << endl;

	cout << endl;
	cout << "pitch: " << pitchPtr.pitch << endl;		//输出实际的宽度值
	cout << "xsize: " << pitchPtr.xsize << endl;		//有效宽度--等于extent.width
	cout << "ysize: " << pitchPtr.ysize << endl;		//有效高度--等于extent.height

	cudaFree(pDeviceData);
	cudaFree(pitchPtr.ptr);
	cin.get();
	return 0;
}

