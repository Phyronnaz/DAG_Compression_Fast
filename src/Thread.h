#pragma once

#include "Core.h"
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <queue>

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
template<typename TInput, typename TResult>
class TThread
{
public:
	TThread(
		const char* Name,
		std::function<TOptional<TInput>()> GetInputFunction,
		std::function<TResult(TInput)> ProcessFunction,
		uint32 ResultsToCache)
		: Name(Name)
		, GetInputFunction(GetInputFunction)
		, ProcessFunction(ProcessFunction)
		, ResultsToCache(ResultsToCache)
	{
		check(ResultsToCache >= 1);
		Threads.emplace_back([this]() { RunThread(); }); // TODO should add ResultsToCache threads
	}
	~TThread()
	{
		check(StopAtomic);
		if (!Joined)
		{
			Join();
		}
	}

	template<int32 Count>
	inline TOptional<TStaticArray<TResult, Count>> GetResult()
	{
		PROFILE_FUNCTION();

		check(ResultsToCache >= Count);
		
		TOptional<TStaticArray<TResult, Count>> Result;
		while (true)
		{
			{
				std::lock_guard<std::mutex> Guard(ResultsMutex);
				if (Results.size() >= Count)
				{
					TStaticArray<TResult, Count> TempResult;
					for (int32 Index = 0; Index < Count; Index++)
					{
						TempResult[Index] = Results.front();
						Results.pop();
					}
					Result = TempResult;
					ProcessWakeup.notify_all();
					break;
				}
			}

			// Do that after checking if there are still some results
			if (StopAtomic)
			{
				check(Results.empty());
				break;
			}

			PROFILE_SCOPE_COLOR(COLOR_YELLOW, "Wait for result");
			const auto NeedToWakeup = [this]()
			{
				if (StopAtomic) return true;
				std::lock_guard<std::mutex> Guard(ResultsMutex);
				return Results.size() >= Count;
			};
			std::unique_lock<std::mutex> Lock(ProcessFinishedMutex);
			ProcessFinished.wait(Lock, NeedToWakeup);
		}
		return Result;
	}
	inline TOptional<TResult> GetSingleResult()
	{
		auto Result = GetResult<1>();
		if (Result.IsValid())
		{
			return Result.GetValue()[0];
		}
		else
		{
			return {};
		}
	}
	
	inline void Join()
	{
		PROFILE_FUNCTION_COLOR(COLOR_YELLOW);

		check(!Joined);
		for (auto& Thread : Threads)
		{
			Thread.join();
		}
		Joined = true;
	}

private:
	void RunThread()
	{
		NAME_THREAD(Name);
		
		while (!StopAtomic)
		{
			bool Wait = true;
			{
				std::lock_guard<std::mutex> Guard(ResultsMutex);
				if (Results.size() + ResultsBeingProcessed < ResultsToCache)
				{
					ResultsBeingProcessed++;
					Wait = false;
				}
			}

			if (Wait)
			{
				const auto ShouldWakeup = [this]()
				{
					if (StopAtomic) return true;
					std::lock_guard<std::mutex> Guard(ResultsMutex);
					return Results.size() + ResultsBeingProcessed < ResultsToCache;
				};
				PROFILE_SCOPE_COLOR(COLOR_YELLOW, "Wait");
				std::unique_lock<std::mutex> Lock(ProcessWakeupMutex);
				ProcessWakeup.wait(Lock, ShouldWakeup);
			}
			else
			{
				TOptional<TInput> NewInput;
				TResult NewResult;

				{
					PROFILE_SCOPE("Get Input");
					NewInput = GetInputFunction();
				}

				if (!NewInput.IsValid())
				{
					StopAtomic = true;
					ProcessWakeup.notify_all();
					ProcessFinished.notify_all();
					break;
				}

				{
					PROFILE_SCOPE("Process");
					NewResult = ProcessFunction(NewInput.GetValue());
				}

				std::lock_guard<std::mutex> Guard(ResultsMutex);
				checkAlways(ResultsBeingProcessed-- >= 0);
				Results.push(NewResult);
				ProcessFinished.notify_one();
			}
		}
	}

	const char* const Name;
	const std::function<TOptional<TInput>()> GetInputFunction;
	const std::function<TResult(TInput)> ProcessFunction;
	const uint32 ResultsToCache;

	std::mutex ResultsMutex;
	int32 ResultsBeingProcessed = 0;
	std::queue<TResult> Results;

	std::mutex ProcessWakeupMutex;
	std::condition_variable ProcessWakeup;

	std::mutex ProcessFinishedMutex;
	std::condition_variable ProcessFinished;
	
	std::vector<std::thread> Threads;
	std::atomic<bool> StopAtomic{ false };

	bool Joined = false;
};

template<typename T>
struct TIndexedData
{
	uint32 Index = 0xFFFFFFFF;
	T Data;
};

template<typename TInput, typename TResult>
class TIndexedThread : public TThread<TIndexedData<TInput>, TIndexedData<TResult>>
{
public:
	TIndexedThread(
		const char* Name,
		std::function<TOptional<TIndexedData<TInput>>()> GetInputFunction,
		std::function<TResult(uint32, TInput)> ProcessFunction,
		uint32 ResultsToCache)
		: TThread<TIndexedData<TInput>, TIndexedData<TResult>>(
			Name,
			GetInputFunction,
			[ProcessFunction](TIndexedData<TInput> Input) { return TIndexedData<TResult>{ Input.Index, ProcessFunction(Input.Index, Input.Data) }; },
			ResultsToCache)
	{
	}
};

template<typename T>
inline auto GetResultLambda(T& Thread)
{
	return [&]() { return Thread.GetSingleResult(); };
}