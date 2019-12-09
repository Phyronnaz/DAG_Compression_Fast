#pragma once

#include "Core.h"

#include <functional>
#include <queue>
#include <atomic>

template<class T>
class TSafeQueue
{
public:
	TSafeQueue() = default;
	~TSafeQueue() = default;

	void Enqueue(T Element)
	{
		FScopeLock Lock(Mutex);
		Queue.push(Element);
	}
	bool Dequeue(T& Element)
	{
		FScopeLock Lock(Mutex);
		if (Queue.empty()) return false;
		Element = Queue.front();
		Queue.pop();
		return true;
	}
	bool IsEmpty() const
	{
		FScopeLock Lock(Mutex);
		return Queue.empty();
	}
	
private:
	std::queue<T> Queue;
	mutable TracyLockable(std::mutex, Mutex)
};

class FThreadPool
{
public:
	using FTask = std::function<void()>;
	
private:
	const char* const Name;
	const FTask InitThread;
	const FTask DestroyThread;
	
	std::vector<std::thread> Threads;
	TSafeQueue<FTask> Tasks;
	std::atomic<uint32> Stop{ 0 };

	TracyLockable(std::mutex, OnEnqueueMutex)
	std::condition_variable_any OnEnqueue;
	
	std::atomic<uint32> WaitingThreads{ 0 };
	TracyLockable(std::mutex, OnAllWaitingMutex)
	std::condition_variable_any OnAllWaiting;
	
public:
	explicit FThreadPool(
		int32 NumThreads,
		const char* Name,
		FTask InitThread = []() {},
		FTask DestroyThread = []() {})
		: Name(Name)
		, InitThread(InitThread)
		, DestroyThread(DestroyThread)
	{
		PROFILE_FUNCTION();
		
		Threads.reserve(NumThreads);
		for (int32 Index = 0; Index < NumThreads; ++Index)
		{
			Threads.emplace_back([this]() { RunThread(); });
		}
	}
	~FThreadPool()
	{
		check(Stop);
	}

	void WaitForCompletion()
	{
		PROFILE_FUNCTION();
		
		if (Threads.empty())
		{
			FTask Task;
			while (Tasks.Dequeue(Task))
			{
				Task();
			}
		}
		else
		{
			const auto ShouldWakeup = [this]() { return Tasks.IsEmpty() && WaitingThreads == Threads.size(); };
			FUniqueLock Lock(OnAllWaitingMutex);
			OnAllWaiting.wait(Lock, ShouldWakeup);
		}
	}
	void Enqueue(FTask Task)
	{
		PROFILE_FUNCTION_TRACY();

		Tasks.Enqueue(std::move(Task));
		OnEnqueue.notify_all();
	}
	void Destroy()
	{
		PROFILE_FUNCTION();

		Stop = 1;
		OnEnqueue.notify_all();
		for (auto& Thread : Threads)
		{
			Thread.join();
		}
	}

private:
	void RunThread()
	{
		PROFILE_FUNCTION();
		NAME_THREAD(Name);

		InitThread();
		
		while (!Stop)
		{
			FTask Task;
			if (Tasks.Dequeue(Task))
			{
				Task();
				CUDA_SYNCHRONIZE_STREAM();
			}
			else
			{
				FUniqueLock Lock(OnEnqueueMutex);
				if (WaitingThreads.fetch_add(1) + 1 == Threads.size())
				{
					OnAllWaiting.notify_all();
				}
				const auto ShouldWakeup = [this]() { return Stop || !Tasks.IsEmpty(); };
				OnEnqueue.wait(Lock, ShouldWakeup);
				WaitingThreads--;
			}
		}

		DestroyThread();
	}
};