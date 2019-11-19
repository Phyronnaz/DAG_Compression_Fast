#pragma once

#include "Core.h"
#include "Array.h"

#include <string>
#include <fstream>
#include <iosfwd>

class FFileWriter
{
public:
	explicit FFileWriter(const std::string& Path)
		: Os(Path, std::ios::binary)
	{
		LOG("Writing to %s", Path.c_str());
		checkfAlways(Os.is_open(), "File: %s", Path.c_str());
	}
	~FFileWriter()
	{
		checkAlways(Os.good());
		Os.close();
		LOG("Writing took %fms", double((std::chrono::high_resolution_clock::now() - StartTime).count()) / 1.e6);
	}
	
	inline void Write(const void* Data, size_t Num)
	{
		Os.write(reinterpret_cast<const char*>(Data), std::streamsize(Num));
		check(Os.good());
	}
	inline void Write(uint32 Data)
	{
		Write(&Data, sizeof(uint32));
	}
	inline void Write(uint64 Data)
	{
		Write(&Data, sizeof(uint64));
	}
	inline void Write(double Data)
	{
		Write(&Data, sizeof(double));
	}
	template<typename T, typename Size = uint64>
	inline void Write(const TStaticArray<T, EMemoryType::CPU, Size>& Array)
	{
		Write(Array.Num());
		Write(Array.GetData(), Array.Num() * sizeof(T));
	}

private:
	const std::chrono::time_point<std::chrono::high_resolution_clock> StartTime = std::chrono::high_resolution_clock::now();
	std::ofstream Os;
};

class FFileReader
{
public:
	explicit FFileReader(const std::string& Path)
		: Is(Path, std::ios::binary)
	{
		LOG("Loading from %s", Path.c_str());
		checkfAlways(Is.is_open(), "File: %s", Path.c_str());
	}
	~FFileReader()
	{
		checkAlways(Is.good());
		Is.close();
		LOG("Loading took %fms", double((std::chrono::high_resolution_clock::now() - StartTime).count()) / 1.e6);
	}

	inline void Read(void* Data, size_t Num)
	{
		Is.read(reinterpret_cast<char*>(Data), std::streamsize(Num));
		check(Is.good());
	}
	inline void Read(uint32& Data)
	{
		Read(&Data, sizeof(uint32));
	}
	inline void Read(uint64& Data)
	{
		Read(&Data, sizeof(uint64));
	}
	inline void Read(double& Data)
	{
		Read(&Data, sizeof(double));
	}
	template<typename T, typename TSize = uint64>
	inline void Read(TStaticArray<T, EMemoryType::CPU, TSize>& Array, const char* Name)
	{
		check(!Array.IsValid());
		TSize Size;
		Read(Size);
		Array = TStaticArray<T, EMemoryType::CPU, TSize>(Name, Size);
		Read(Array.GetData(), Array.Num() * sizeof(T));
	}

private:
	const std::chrono::time_point<std::chrono::high_resolution_clock> StartTime = std::chrono::high_resolution_clock::now();
	std::ifstream Is;
};