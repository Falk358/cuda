#include <stdio.h>
int main()
{
	int deviceCount;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	for (int i = 0; i < deviceCount; ++i)
	{
		cudaSetDevice(i);
		cudaDeviceProp deviceProp;
		int driverVersion;
		int runtimeVersion;
		cudaError_t error_id_driver = cudaDriverGetVersion(&driverVersion);
		cudaError_t error_id_runtime = cudaRuntimeGetVersion(&runtimeVersion);
		cudaGetDeviceProperties(&deviceProp, i);
		printf("\nDevice %d: %s\n", i, deviceProp.name);
		printf("	Driver Version %d\n 	Runtime Version: %d\n 	GPU Memory in MB: %d\n 	Max Num Threads per block: %d\n 	UVA available: %d", driverVersion, runtimeVersion, deviceProp.totalGlobalMem/1024/1024, deviceProp.maxThreadsPerBlock, deviceProp.unifiedAddressing);
		
	}


}

