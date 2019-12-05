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
		std::lock_guard<std::mutex> Lock(Mutex);
		Queue.push(Element);
	}
	bool Dequeue(T& Element)
	{
		std::unique_lock<std::mutex> Lock(Mutex);
		if (Queue.empty()) return false;
		Element = Queue.front();
		Queue.pop();
		return true;
	}
	
private:
	std::queue<T> Queue;
	mutable std::mutex Mutex;
};

class FThreadPool
{
public:
	using FTask = std::function<void()>;
	
private:
	std::vector<std::thread> Threads;
	TSafeQueue<FTask> Tasks;
	std::atomic<uint32> Stop{ 0 };

	std::mutex OnEnqueueMutex;
	std::condition_variable OnEnqueue;
	
	std::atomic<uint32> WaitingThreads{ 0 };
	std::mutex OnAllWaitingMutex;
	std::condition_variable OnAllWaiting;
	
public:
	explicit FThreadPool(int32 NumThreads)
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
			std::unique_lock<std::mutex> Lock(OnAllWaitingMutex);
			OnAllWaiting.wait(Lock);
		}
	}
	void Enqueue(FTask Task)
	{
		PROFILE_FUNCTION();

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
		NAME_THREAD("ThreadPool");
		
		while (!Stop)
		{
			FTask Task;
			if (Tasks.Dequeue(Task))
			{
				Task();
			}
			else
			{
				std::unique_lock<std::mutex> Lock(OnEnqueueMutex);
				if (1 + WaitingThreads.fetch_add(1) == Threads.size())
				{
					OnAllWaiting.notify_all();
				}
				OnEnqueue.wait(Lock);
				WaitingThreads--;
			}
		}
	}
};