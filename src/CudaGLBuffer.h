#pragma once

#include "Core.h"
#include "GL/glew.h"
#include <cuda_gl_interop.h>

class FCudaGLBuffer
{
public:
	cudaGraphicsResource_t CudaResource = nullptr;
	cudaSurfaceObject_t CudaSurface = 0;

	FCudaGLBuffer() = default;

	void RegisterResource(const GLuint Image)
	{
		check(!CudaResource);
		CUDA_CHECKED_CALL cudaGraphicsGLRegisterImage(&CudaResource, Image, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		check(CudaResource);
	}
	void UnregisterResource()
	{
		check(CudaResource);
		CUDA_CHECKED_CALL cudaGraphicsUnregisterResource(CudaResource);
		CudaResource = nullptr;
	}
	void MapSurface()
	{
		check(!CudaSurface);
		CUDA_CHECKED_CALL cudaGraphicsMapResources(1, &CudaResource);
		cudaArray_t cudaArray;
		CUDA_CHECKED_CALL cudaGraphicsSubResourceGetMappedArray(&cudaArray, CudaResource, 0, 0);
		cudaResourceDesc cudaArrayResourceDesc;
		memset(&cudaArrayResourceDesc, 0, sizeof(cudaArrayResourceDesc));
		cudaArrayResourceDesc.resType = cudaResourceTypeArray;
		cudaArrayResourceDesc.res.array.array = cudaArray;
		CUDA_CHECKED_CALL cudaCreateSurfaceObject(&CudaSurface, &cudaArrayResourceDesc);
		check(CudaSurface);
	}
	void UnmapSurface()
	{
		check(CudaSurface);
		CUDA_CHECKED_CALL cudaDestroySurfaceObject(CudaSurface);
		CUDA_CHECKED_CALL cudaGraphicsUnmapResources(1, &CudaResource);
		CudaSurface = 0;
	};
};