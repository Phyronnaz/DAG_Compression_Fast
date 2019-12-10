#pragma once

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

#define ENABLE_CHECKS 1
#define DEBUG_GPU_ARRAYS 0

#define ENABLE_COLORS 0
#define TOP_LEVEL MAX(LEVELS - 8, 0)

#define LEVELS 15
#define SUBDAG_LEVELS 12
#define SUBDAG_MAX_NUM_FRAGMENTS (150 * 1024 * 1024)

#define VOXELIZER_LEVELS 10
#define VOXELIZER_MAX_NUM_FRAGMENTS (10 * 1024 * 1024) // Needs to be as small as possible: larger buffer size leads to 10x slower rasterization


#define GPU_MEMORY_ALLOCATED_IN_GB 6.0

#define MERGE_MIN_NODES_FOR_GPU 50000 // Under this number of nodes in a level, the CPU will be used

#define NUM_MERGE_COLORS_THREADS 4


#define NUM_MERGE_THREADS 0
#define PRINT_DEBUG_INFO 0
#define DEBUG_THRUST 0

#define ENABLE_FORCEINLINE !ENABLE_CHECKS
#define ENABLE_FLATTEN 0

static_assert(SUBDAG_LEVELS <= LEVELS, "");
static_assert(!DEBUG_GPU_ARRAYS || NUM_MERGE_THREADS == 0 || LEVELS == SUBDAG_LEVELS, "");

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER

#if defined(__CUDA_ARCH__)
#define DEBUG_BREAK() __threadfence(); asm("trap;")
#else
#define DEBUG_BREAK() __debugbreak()
#endif
#define ASSUME(expr) check(expr); __assume(expr);

#define _CRT_SECURE_NO_WARNINGS

#if ENABLE_CHECKS
#undef NDEBUG
#endif

#else

#include <csignal>

#if defined(__CUDA_ARCH__)
#define DEBUG_BREAK() __threadfence(); asm("trap;")
#else
#define DEBUG_BREAK() raise(SIGTRAP)
#endif
#define ASSUME(expr)

#endif

#define RESTRICT __restrict__

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include <cinttypes> // PRIu64
#include <cstdint>
#include <cstdio> // fprintf
#include <cstdlib> // abort

using int8 = std::int8_t;
using int16 = std::int16_t;
using int32 = std::int32_t;
using int64 = std::int64_t;

using uint8 = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using uint64 = std::uint64_t;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#if defined(__INTELLISENSE__) || defined(__RSCPP_VERSION) || defined(__CLION_IDE__) || defined(__JETBRAINS_IDE__)
#define __PARSER__
#endif

#ifdef __PARSER__
#error "compiler detected as parser"

#define __CUDACC__
#define __CUDA_ARCH__
#include "crt/math_functions.h"
#include "crt/device_functions.h"
#include "vector_functions.h"

#define __host__
#define __device__
#define __global__
#define __device_builtin__
#endif

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime.h>
#include "cnmem.h"

#if ENABLE_CHECKS
#define checkf(expr, msg, ...) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s: " msg "\n", __LINE__, #expr, ##__VA_ARGS__); DEBUG_BREAK(); }
#define check(expr) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s\n", __LINE__, #expr); DEBUG_BREAK(); }
#define checkAlways(expr) check(expr)
#define checkfAlways(expr, msg, ...) checkf(expr,msg,##__VA_ARGS__)

#define checkEqual(a, b) checkf((a) == (b), "As uint64: a = %" PRIu64 "; b = %" PRIu64, uint64(a), uint64(b))
#define checkInf(a, b) checkf((a) < (b), "As uint64: a = %" PRIu64 "; b = %" PRIu64, uint64(a), uint64(b))
#define checkInfEqual(a, b) checkf((a) <= (b), "As uint64: a = %" PRIu64 "; b = %" PRIu64, uint64(a), uint64(b))
#else
#define checkf(expr, msg, ...)
#define check(...)
#define checkAlways(expr) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s\n", __LINE__, #expr); DEBUG_BREAK(); std::abort(); }
#define checkfAlways(expr, msg, ...) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s: " msg "\n", __LINE__, #expr,##__VA_ARGS__); DEBUG_BREAK(); std::abort(); }
#define checkEqual(a, b)
#define checkInf(a, b)
#define checkInfEqual(a, b)
#endif

#define LOG(msg, ...) printf(msg "\n",##__VA_ARGS__)

#if PRINT_DEBUG_INFO
#define LOG_DEBUG(msg, ...) LOG(msg, __VA_ARGS__)
#else
#define LOG_DEBUG(msg, ...) if(0) { LOG(msg, __VA_ARGS__); }
#endif

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#if ENABLE_FORCEINLINE
#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#else
#define FORCEINLINE __attribute__((always_inline))
#endif
#else
#define FORCEINLINE
#endif

#if ENABLE_FLATTEN
#ifdef _MSC_VER
#define FLATTEN
#else
#define FLATTEN __attribute__((flatten))
#endif
#else
#define FLATTEN
#endif

#define HOST           __host__ inline FORCEINLINE FLATTEN
#define HOST_RECURSIVE __host__ inline

#define DEVICE           __device__ inline FORCEINLINE FLATTEN
#define DEVICE_RECURSIVE __device__ inline

#define HOST_DEVICE           __host__ __device__ inline FORCEINLINE FLATTEN
#define HOST_DEVICE_RECURSIVE __host__ __device__ inline

#define GPU_LAMBDA __host__ __device__
#define GPU_ONLY_LAMBDA __device__

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
namespace detail
{
	struct CudaErrorChecker
	{
		const int32 Line;
		const char* File;

		inline CudaErrorChecker(int32 Line, const char* File) noexcept
			: Line(Line)
			, File(File)
		{
		}

		FORCEINLINE cudaError_t operator=(cudaError_t Error) const
		{
			if (Error != cudaSuccess)
			{
				std::fprintf(stderr, "ERROR %s:%d: CUDA error \"%s\"\n", File, Line, cudaGetErrorString(Error));
				std::abort();
			}
			return Error;
		}

		FORCEINLINE CUresult operator=(CUresult Error) const
		{
			if (Error != CUDA_SUCCESS) 
			{
				const char* Message = nullptr;
				cuGetErrorString(Error, &Message);
				std::fprintf(stderr, "ERROR [driver] %s:%d: CUDA error \"%s\"\n", File, Line, Message);
				std::abort();
			}

			return Error;
		}

		FORCEINLINE cnmemStatus_t operator=(cnmemStatus_t Error) const
		{
			if (Error != CNMEM_STATUS_SUCCESS) 
			{
				const char* Message = cnmemGetErrorString(Error);
				std::fprintf(stderr, "ERROR %s:%d: CNMEM error \"%s\"\n", File, Line, Message);
				std::abort();
			}

			return Error;
		}
	};
}

#include <limits>

template<typename T, typename U>
HOST_DEVICE T Cast(U Value)
{
	static_assert(std::numeric_limits<T>::max() <= std::numeric_limits<int64>::max(), "invalid cast");
	checkf(
		int64(std::numeric_limits<T>::min()) <= int64(Value) && int64(Value) <= int64(std::numeric_limits<T>::max()),
		"overflow: %" PRIi64 " not in [%" PRIi64 ", %" PRIi64 "]",
		int64(Value),
		int64(std::numeric_limits<T>::min()), int64(std::numeric_limits<T>::max()));
	return static_cast<T>(Value);
}

#pragma warning ( push )
#pragma warning ( disable: 4099 4456 4505 )
#include "tracy/Tracy.hpp"
#pragma warning ( pop )

#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"

#ifdef TRACY_ENABLE
#define PROFILE_SCOPE_COLOR_TRACY(Color, Format, ...) \
	ZoneScopedC(Color); \
	{ \
		char __String[1024]; \
		const int32 __Size = sprintf(__String, Format, ##__VA_ARGS__); \
		checkInfEqual(__Size, 1024); \
		ZoneName(__String, __Size); \
	}

#define ZONE_METADATA_TRACY(Format, ...) \
	{ \
		char __String[1024]; \
		const int32 __Size = sprintf(__String, Format, ##__VA_ARGS__); \
		checkInfEqual(__Size, 1024); \
		ZoneText(__String, __Size); \
	}

#define MARK_TRACY(Name) FrameMarkNamed(Name)
#define NAME_THREAD_TRACY(Name) tracy::SetThreadName(Name);
#else
#define PROFILE_SCOPE_COLOR_TRACY(Color, Format, ...)
#define ZONE_METADATA_TRACY(Format, ...)
#define MARK_TRACY(Name)
#define NAME_THREAD_TRACY(Name)
#endif

struct FNVExtScope
{
	template<typename... TArgs>
	FNVExtScope(uint32 Color, const char* Format, TArgs... Args)
	{
		const int32 Size = sprintf(String, Format, Args...);
		checkInfEqual(Size, 1024);
		
		eventAttrib.version = NVTX_VERSION;
		eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		eventAttrib.colorType = NVTX_COLOR_ARGB;
		eventAttrib.color = Color;
		eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
		eventAttrib.message.ascii = String;
		nvtxRangePushEx(&eventAttrib);
	}
	~FNVExtScope()
	{
		nvtxRangePop();
	}
	
    nvtxEventAttributes_t eventAttrib = {0};
	char String[1024];
};

#define JOIN_IMPL(A, B) A ## B
#define JOIN(A, B) JOIN_IMPL(A, B)

#define COLOR_BLACK 0x000000
#define COLOR_RED 0xFF0000
#define COLOR_GREEN 0x00FF00
#define COLOR_BLUE 0x0000FF
#define COLOR_YELLOW 0xFFFF00

#define PROFILE_SCOPE_COLOR_NV(Color, Format, ...) FNVExtScope JOIN(ScopePerf, __LINE__)(Color, Format, ##__VA_ARGS__);

#define MARK_NV(Name) nvtxMarkA(Name)
#define NAME_THREAD_NV(Name) nvtxNameOsThreadA(GetCurrentThreadId(), Name);

#define PROFILE_SCOPE_COLOR(Color, Format, ...) PROFILE_SCOPE_COLOR_NV(Color, Format, ##__VA_ARGS__); PROFILE_SCOPE_COLOR_TRACY(Color, Format, ##__VA_ARGS__);
#define PROFILE_SCOPE(Format, ...) PROFILE_SCOPE_COLOR_NV(COLOR_RED, Format, ##__VA_ARGS__); PROFILE_SCOPE_COLOR_TRACY(COLOR_BLACK, Format, ##__VA_ARGS__);

#define PROFILE_SCOPE_TRACY(Format, ...) PROFILE_SCOPE_COLOR_TRACY(COLOR_BLACK, Format, ##__VA_ARGS__);
#define PROFILE_FUNCTION_TRACY() PROFILE_SCOPE_TRACY(__FUNCTION__);
#define PROFILE_FUNCTION_COLOR_TRACY(Color) PROFILE_SCOPE_COLOR_TRACY(Color, __FUNCTION__);

#define PROFILE_FUNCTION_COLOR(Color) PROFILE_SCOPE_COLOR(Color, __FUNCTION__);
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__);

#define MARK(Name) MARK_NV(Name); MARK_TRACY(Name);
#define NAME_THREAD(Name) NAME_THREAD_NV(Name); NAME_THREAD_TRACY(Name);

#define ZONE_METADATA(Format, ...) ZONE_METADATA_TRACY(Format, ##__VA_ARGS__)

#include <mutex>

using FScopeLock = std::lock_guard<LockableBase(std::mutex)>;
using FUniqueLock = std::unique_lock<LockableBase(std::mutex)>;

#define DEFAULT_STREAM GetDefaultStream()

#define CUDA_CHECKED_CALL ::detail::CudaErrorChecker(__LINE__,__FILE__) =
#define CUDA_SYNCHRONIZE_STREAM() CUDA_CHECKED_CALL StreamSynchronize();
#define CUDA_CHECK_LAST_ERROR() CUDA_CHECKED_CALL cudaGetLastError(); CUDA_SYNCHRONIZE_STREAM()

inline auto GetDefaultStream()
{
	thread_local cudaStream_t Stream = nullptr;
	if (!Stream)
	{
		CUDA_CHECKED_CALL cudaStreamCreate(&Stream);
	}
	check(Stream);
	return Stream;
}

inline auto StreamSynchronize()
{
	PROFILE_FUNCTION_TRACY();
	return cudaStreamSynchronize(DEFAULT_STREAM);
}