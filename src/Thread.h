#pragma once

#include "Core.h"
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <queue>
#include <unordered_set>

template<typename T>
struct TOptional
{
	TOptional() = default;
	TOptional(T Value)
		: Value(Value)
		, HasValue(true)
	{
	}

	inline bool IsValid() const
	{
		return HasValue;
	}
	inline T GetValue() const
	{
		check(IsValid());
		return Value;
	}
	
private:
	T Value;
	bool HasValue = false;
};

struct FEmpty
{
};

// Will loop until GetInputFunction returns an invalid input
template<typename TInput, typename TResult, typename TQuery>
class TThread
{
public:
	TThread(
		const char* Name,
		std::function<TInput(TQuery)> GetInputFunction,
		std::function<TResult(TQuery, TInput)> ProcessFunction,
		uint32 NumThreads,
		std::function<void()> InitThread = []() {},
		std::function<void()> DestroyThread = []() {})
		: Name(Name)
		, GetInputFunction(GetInputFunction)
		, ProcessFunction(ProcessFunction)
		, InitThread(InitThread)
		, DestroyThread(DestroyThread)
	{
		check(NumThreads >= 1);
		for (uint32 Index = 0; Index < NumThreads; Index++)
		{
			Threads.emplace_back([this, Index]() { RunThread(Index); });
		}
	}
	~TThread()
	{
		check(QueriesToProcess.empty());
		check(ProcessedQueries.empty());
		if (!StopAtomic)
		{
			Stop();
		}
		if (!Joined)
		{
			Join();
		}
	}

	inline TResult GetResult_DoNotStartQuery(TQuery Query)
	{
		PROFILE_FUNCTION_COLOR_TRACY(COLOR_YELLOW);

		TOptional<TResult> Result;
		
		const auto Check = [&]()
		{
			check(!StopAtomic);
			FScopeLock Lock(ProcessedQueriesMutex);
			const auto ResultPtr = ProcessedQueries.find(Query);
			if (ResultPtr != ProcessedQueries.end())
			{
				Result = ResultPtr->second;
				ProcessedQueries.erase(ResultPtr);
			}
			return Result.IsValid();
		};

		FUniqueLock Lock(ProcessFinishedMutex);
		while (!Check()) ProcessFinished.wait_for(Lock, std::chrono::milliseconds(10));
		
		return Result.GetValue();
	}
	template<uint32 Count>
	inline void StartQueries(TStaticArray<TQuery, Count> Queries)
	{
		FScopeLock Lock(QueriesToProcessMutex);
		for (auto& Query : Queries)
		{
			if (AlreadyQueuedQueries.find(Query) == AlreadyQueuedQueries.end())
			{
				AlreadyQueuedQueries.insert(Query);
				QueriesToProcess.push(Query);
			}
		}
		ProcessWakeup.notify_all();
	}
	inline TResult GetResult(TQuery Query)
	{
		PROFILE_FUNCTION_COLOR_TRACY(COLOR_YELLOW);

		StartQueries<1>({ Query });

		return GetResult_DoNotStartQuery(Query);
	}
	// Needed to make all threads work before waiting
	template<uint32 Count>
	inline TStaticArray<TResult, Count> GetResults(TStaticArray<TQuery, Count> Queries)
	{
		PROFILE_FUNCTION_COLOR_TRACY(COLOR_YELLOW);

		StartQueries(Queries);

		TStaticArray<TResult, Count> Results;
		for(uint32 Index = 0; Index < Count; Index++)
		{
			Results[Index] = GetResult_DoNotStartQuery(Queries[Index]);
		}
		return Results;
	}

	inline void Stop()
	{
		check(!StopAtomic);
		StopAtomic = true;
	}
	inline void Join()
	{
		PROFILE_FUNCTION_COLOR_TRACY(COLOR_YELLOW);

		check(!Joined);
		check(StopAtomic);
		for (auto& Thread : Threads)
		{
			Thread.join();
		}
		Joined = true;
	}

private:
	void RunThread(int32 ThreadIndex)
	{
		char Buffer[1024];
		sprintf(Buffer, "%s #%d", Name, ThreadIndex);
		NAME_THREAD(Buffer);

		InitThread();
		
		while (!StopAtomic)
		{
			TOptional<TQuery> OptionalQuery;
			{
				FScopeLock Lock(QueriesToProcessMutex);
				if (!QueriesToProcess.empty())
				{
					OptionalQuery = QueriesToProcess.front();
					QueriesToProcess.pop();
				}
			}

			if (OptionalQuery.IsValid())
			{
				TQuery Query = OptionalQuery.GetValue();
				TInput NewInput;
				TResult NewResult;

				{
					PROFILE_SCOPE("Get Input %u", Query);
					NewInput = GetInputFunction(Query);
				}
				{
					PROFILE_SCOPE("Process %u", Query);
					NewResult = ProcessFunction(Query, NewInput);
					CUDA_SYNCHRONIZE_STREAM();
				}

				FScopeLock Lock(ProcessedQueriesMutex);
				check(ProcessedQueries.find(Query) == ProcessedQueries.end());
				ProcessedQueries[Query] = NewResult;
				ProcessFinished.notify_one();
			}
			else
			{
				const auto ShouldWakeup = [this]()
				{
					if (StopAtomic) return true;
					FScopeLock Lock(QueriesToProcessMutex);
					return !QueriesToProcess.empty();
				};
				PROFILE_SCOPE_COLOR_TRACY(COLOR_YELLOW, "Wait");
				FUniqueLock Lock(ProcessWakeupMutex);
				while (!ShouldWakeup()) ProcessWakeup.wait_for(Lock, std::chrono::milliseconds(10));
			}
		}

		DestroyThread();

		CUDA_CHECKED_CALL cudaStreamDestroy(DEFAULT_STREAM);
	}

	const char* const Name;
	const std::function<TInput(TQuery)> GetInputFunction;
	const std::function<TResult(TQuery, TInput)> ProcessFunction;
	const std::function<void()> InitThread;
	const std::function<void()> DestroyThread;

	TracyLockable(std::mutex, QueriesToProcessMutex);
	std::queue<TQuery> QueriesToProcess;
	std::unordered_set<TQuery> AlreadyQueuedQueries;

	TracyLockable(std::mutex, ProcessedQueriesMutex);
	std::unordered_map<TQuery, TResult> ProcessedQueries;
	
	TracyLockable(std::mutex, ProcessWakeupMutex);
	std::condition_variable_any ProcessWakeup;

	TracyLockable(std::mutex, ProcessFinishedMutex);
	std::condition_variable_any ProcessFinished;
	
	std::vector<std::thread> Threads;
	std::atomic<bool> StopAtomic{ false };

	bool Joined = false;
};

template<typename TInput, typename TResult>
using TIndexedThread = TThread<TInput, TResult, uint32>;

template<typename T>
inline auto GetResultLambda(T& Thread)
{
	return [&](auto Query) { return Thread.GetResult(Query); };
}