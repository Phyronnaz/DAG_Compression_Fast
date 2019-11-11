#pragma once

#include "Core.h"
#include "Array.h"

namespace DAGCompression
{
	TStaticArray<uint32, EMemoryType::GPU> CreateDAG(TStaticArray<uint64, EMemoryType::GPU> Fragments, int32 Depth);
}