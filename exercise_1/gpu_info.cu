#include <stdio.h>
int main()
{
	int deviceCount;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	for (int i = 0; i < deviceCount; ++i)
	{
		cudaSetDevice(i);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("\nDevice %d: %s", i, deviceProp.name);	
	}


}
