#pragma once

#include "Core.h"
#include "Array.h"

namespace DAGCompression
{
	TStaticArray<uint32, EMemoryType::GPU> CreateDAG(const TStaticArray<uint64, EMemoryType::GPU>& Fragments, int32 Depth);
}