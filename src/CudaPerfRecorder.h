#pragma once

#include "Core.h"

class FCudaPerfRecorder
{
public:
	FCudaPerfRecorder()
	{
		CUDA_CHECKED_CALL cudaEventCreate(&BeginEvent);
		CUDA_CHECKED_CALL cudaEventCreate(&EndEvent);
	}
	~FCudaPerfRecorder()
	{
		CUDA_CHECKED_CALL cudaEventDestroy(BeginEvent);
		CUDA_CHECKED_CALL cudaEventDestroy(EndEvent);
	}

	inline void Start()
	{
		check(!bStarted);
		bStarted = true;
		CUDA_CHECKED_CALL cudaEventRecord(BeginEvent);
	}
	inline double End()
	{
		CUDA_CHECKED_CALL cudaEventRecord(EndEvent);
		CUDA_CHECKED_CALL cudaEventSynchronize(EndEvent);

		check(bStarted);
		bStarted = false;

		float Elapsed; // in ms
		CUDA_CHECKED_CALL cudaEventElapsedTime(&Elapsed, BeginEvent, EndEvent);
		CUDA_CHECK_ERROR();

		return Elapsed / 1000.;
	}

private:
	bool bStarted = false;
	cudaEvent_t BeginEvent = nullptr;
	cudaEvent_t EndEvent = nullptr;
};

class FCudaScopePerfRecorder
{
public:
	FCudaScopePerfRecorder(const char* Name, uint64 NumElements)
		: Name(Name)
		, NumElements(NumElements)
	{
		PerfRecorder.Start();
	}
	~FCudaScopePerfRecorder()
	{
		const auto Elapsed = PerfRecorder.End();
		LOG_DEBUG("\t%s: Processing %llu elements took %fs: %f B/s", Name, NumElements, Elapsed, NumElements / Elapsed / 1e9);
	}

private:
	const char* const Name;
	const uint64 NumElements;
	FCudaPerfRecorder PerfRecorder;
};