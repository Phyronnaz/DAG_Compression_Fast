#pragma once

#include "Core.h"
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <queue>

template<typename TInput, typename TResult>
class TThread
{
public:
	TThread(
		const std::function<TInput()>& GetInputFunction,
		const std::function<TResult(TInput)>& ProcessFunction,
		int32 ResultsToCache)
		: GetInputFunction(GetInputFunction)
		, ProcessFunction(ProcessFunction)
		, ResultsToCache(ResultsToCache)
		, Thread([this]() { RunThread(); })
	{
		check(ResultsToCache >= 1);
	}

	inline TResult GetResult()
	{
		PROFILE_FUNCTION_COLOR(COLOR_YELLOW);
		
		TResult Result;
		bool HasResult = false;

		while (!HasResult)
		{
			{
				std::lock_guard<std::mutex> Guard(ResultsMutex);
				if (!Results.empty())
				{
					Result = std::move(Results.front());
					Results.pop();
					HasResult = true;
					ProcessWakeup.notify_all(); // One is probably enough, let's wakeup all anyways
				}
			}
			if (!HasResult)
			{
				std::unique_lock<std::mutex> Lock(ProcessFinishedMutex);
				ProcessFinished.wait(Lock);
			}
			check(!StopAtomic);
		}
		check(HasResult);

		return Result;
	}

	inline void Stop()
	{
		check(!StopAtomic.load());
		StopAtomic.store(true);
		ProcessWakeup.notify_all();
		ProcessFinished.notify_all(); // So that the assert is reached
	}
	inline void Join()
	{
		PROFILE_FUNCTION_COLOR(COLOR_YELLOW);
		
		check(StopAtomic.load());
		Thread.join();
	}

private:
	void RunThread()
	{
		while (!StopAtomic)
		{
			bool NeedProcess = false;
			{
				std::lock_guard<std::mutex> Guard(ResultsMutex);
				if (Results.size() + ResultsBeingProcessed < ResultsToCache)
				{
					NeedProcess = true;
					ResultsBeingProcessed++;
				}
			}

			if (NeedProcess)
			{
				TInput NewInput;
				TResult NewResult;

				{
					PROFILE_SCOPE("Get Input");
					NewInput = std::move(GetInputFunction());
				}
				{
					PROFILE_SCOPE("Process");
					NewResult = std::move(ProcessFunction(std::move(NewInput)));
				}

				std::lock_guard<std::mutex> Guard(ResultsMutex);
				checkAlways(ResultsBeingProcessed-- >= 0);
				Results.push(std::move(NewResult));
				ProcessFinished.notify_one();
			}
			else
			{
				PROFILE_SCOPE_COLOR(COLOR_YELLOW, "Wait");
				std::unique_lock<std::mutex> Lock(ProcessWakeupMutex);
				ProcessWakeup.wait(Lock);
			}
		}
	}

	const std::function<TInput()> GetInputFunction;
	const std::function<TResult(TInput)> ProcessFunction;
	const int32 ResultsToCache;

	std::mutex ResultsMutex;
	int32 ResultsBeingProcessed = 0;
	std::queue<TResult> Results;

	std::mutex ProcessWakeupMutex;
	std::condition_variable ProcessWakeup;

	std::mutex ProcessFinishedMutex;
	std::condition_variable ProcessFinished;
	
	std::thread Thread;
	std::atomic<bool> StopAtomic{ false };
};