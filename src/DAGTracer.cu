#include "DAGTracer.h"

FDAGTracer::FDAGTracer(uint32 Width, uint32 Height)
	: Width(Width)
	, Height(Height)
{
	glGenTextures(1, &ColorsImage);
	glBindTexture(GL_TEXTURE_2D, ColorsImage);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int32)Width, (int32)Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
	ColorsBuffer.RegisterResource(ColorsImage);
}

FDAGTracer::~FDAGTracer()
{
	ColorsBuffer.UnregisterResource();
	glDeleteTextures(1, &ColorsImage);
}

DEVICE uint32 float3_to_rgb888(float3 c)
{
	float R = fmin(1.0f, fmax(0.0f, c.x));
	float G = fmin(1.0f, fmax(0.0f, c.y));
	float B = fmin(1.0f, fmax(0.0f, c.z));
	return	(uint32_t(R * 255.0f) << 0) |
		(uint32_t(G * 255.0f) << 8) |
		(uint32_t(B * 255.0f) << 16);
}

// order: (shouldFlipX, shouldFlipY, shouldFlipZ)
DEVICE uint8 next_child(uint8 order, uint8 mask)
{
	for (uint8 child = 0; child < 8; ++child)
	{
		uint8 childInOrder = child ^ order;
		if (mask & (1u << childInOrder))
			return childInOrder;
	}
	check(false);
	return 0;
}

struct FPath
{
public:
	uint3 path;
	
	HOST_DEVICE FPath(uint3 path)
		: path(path)
	{
	}
	HOST_DEVICE FPath(uint32 x, uint32 y, uint32 z)
		: path(make_uint3(x, y, z))
	{
	}

	HOST_DEVICE void ascend(uint32 levels)
	{
		path.x >>= levels;
		path.y >>= levels;
		path.z >>= levels;
	}
	HOST_DEVICE void descend(uint8 child)
	{
		path.x <<= 1;
		path.y <<= 1;
		path.z <<= 1;
		path.x |= (child & 0x4u) >> 2;
		path.y |= (child & 0x2u) >> 1;
		path.z |= (child & 0x1u) >> 0;
	}

	// level: level of the child!
	HOST_DEVICE uint8 child_index(uint32 level, uint32 totalLevels) const
	{
		check(level <= totalLevels);
		return uint8(
			(((path.x >> (totalLevels - level) & 0x1) == 0) ? 0 : 4) |
			(((path.y >> (totalLevels - level) & 0x1) == 0) ? 0 : 2) |
			(((path.z >> (totalLevels - level) & 0x1) == 0) ? 0 : 1));
	}

	HOST_DEVICE float3 as_position(uint32 extraShift = 0) const
	{
		return make_float3(
			float(path.x << extraShift),
			float(path.y << extraShift),
			float(path.z << extraShift)
		);
	}
};

struct Leaf
{
	uint32 low;
	uint32 high;

	HOST_DEVICE uint64 to_64() const
	{
		return (uint64(high) << 32) | uint64(low);
	}

	HOST_DEVICE uint8 get_first_child_mask() const
	{
#if defined(__CUDA_ARCH__)
		const uint32 u0 = low | (low >> 1);
		const uint32 u1 = u0 | (u0 >> 2);
		const uint32 u2 = u1 | (u1 >> 4);
		const uint32 t0 = u2 & 0x01010101u;

		const uint32 v0 = high | (high >> 1);
		const uint32 v1 = v0 | (v0 >> 2);
		const uint32 v2 = v1 | (v1 << 4); // put into second nibble
		const uint32 s0 = v2 & 0x10101010u;

		uint32 s1;
		asm(
			"{\n\t"
			"    .reg .u32 t0;\n\t"
			"    mul.lo.u32 t0, %1, 0x1020408;\n\t" // multiply keeping the low part
			"    mad.lo.u32 %0, %2, 0x1020408, t0;\n\t" // multiply keeping the low part and add
			"}\n\t"
			: "=r"(s1) : "r"(t0), "r"(s0)
		);

		return s1 >> 24;
#else
		const uint64 currentLeafMask = to_64();
		return uint8(
			((currentLeafMask & 0x00000000000000FF) == 0 ? 0 : 1 << 0) |
			((currentLeafMask & 0x000000000000FF00) == 0 ? 0 : 1 << 1) |
			((currentLeafMask & 0x0000000000FF0000) == 0 ? 0 : 1 << 2) |
			((currentLeafMask & 0x00000000FF000000) == 0 ? 0 : 1 << 3) |
			((currentLeafMask & 0x000000FF00000000) == 0 ? 0 : 1 << 4) |
			((currentLeafMask & 0x0000FF0000000000) == 0 ? 0 : 1 << 5) |
			((currentLeafMask & 0x00FF000000000000) == 0 ? 0 : 1 << 6) |
			((currentLeafMask & 0xFF00000000000000) == 0 ? 0 : 1 << 7));
#endif
	}
	HOST_DEVICE uint8 get_second_child_mask(uint8 firstChildIndex)
	{
		const uint8 shift = uint8((firstChildIndex & 3) * 8);
		const uint32 leafMask = (firstChildIndex & 4) ? high : low;
		return uint8(leafMask >> shift);
	}
};

struct FDAG
{
	TGpuArray<uint32> data;
	
    constexpr static uint32 levels = LEVELS;

	HOST_DEVICE constexpr uint32 leaf_level() const
	{
		return levels - 2;
	}
	HOST_DEVICE constexpr bool is_leaf(uint32 level) const
	{
		return level == leaf_level();
	}

	HOST_DEVICE bool is_valid() const
	{
		return data.IsValid();
	}

	HOST_DEVICE uint32 get_node(uint32 /*level*/, uint32 index) const
	{
		return data[index];
	}
	HOST_DEVICE uint32 get_child_index(uint32 /*level*/, uint32 index, uint8 childMask, uint8 child) const
	{
		return data[index + Utils::ChildOffset(childMask, child)];
	}
	HOST_DEVICE Leaf get_leaf(uint32 index) const
	{
		return { data[index], data[index + 1] };
	}
};

struct FColors
{
	TGpuArray<uint32> colors;
	TGpuArray<uint64> enclosedLeaves;
	
#if ENABLE_COLORS
	HOST_DEVICE uint64 get_leaves_count(uint32 level, uint32 node) const
	{
		const uint32 upperBits = node >> 8;

		// If we are in the top-levels, the obtained value is an index into an 
		// external array
		if (level < TOP_LEVEL)
		{
			return enclosedLeaves[upperBits];
		}
		else
		{
			return upperBits;
		}
	}
	HOST_DEVICE uint32 get_color(uint64 leafIndex) const
	{
		return colors[leafIndex];
	}
#endif
};

template<bool isRoot>
DEVICE uint8 compute_intersection_mask(
	uint32 level,
	const FPath& path,
	const FDAG& dag,
	const float3& rayOrigin,
	const float3& rayDirection,
	const float3& rayDirectionInverted)
{
	// Find node center = .5 * (boundsMin + boundsMax) + .5f
	const uint32 shift = dag.levels - level;

	const float radius = float(1u << (shift - 1));
	const float3 center = make_float3(radius) + path.as_position(shift);

	const float3 centerRelativeToRay = center - rayOrigin;

	// Ray intersection with axis-aligned planes centered on the node
	// => rayOrg + tmid * rayDir = center
	const float3 tmid = centerRelativeToRay * rayDirectionInverted;

	// t-values for where the ray intersects the slabs centered on the node
	// and extending to the side of the node
	float tmin, tmax;
	{
		const float3 slabRadius = radius * abs(rayDirectionInverted);
		const float3 pmin = tmid - slabRadius;
		tmin = max(max(pmin), .0f);

		const float3 pmax = tmid + slabRadius;
		tmax = min(pmax);
	}

	// Check if we actually hit the root node
	// This test may not be entirely safe due to float precision issues.
	// especially on lower levels. For the root node this seems OK, though.
	if (isRoot && (tmin >= tmax))
	{
		return 0;
	}

	// Identify first child that is intersected
	// NOTE: We assume that we WILL hit one child, since we assume that the
	//       parents bounding box is hit.
	// NOTE: To safely get the correct node, we cannot use o+ray_tmin*d as the
	//       intersection point, since this point might lie too close to an
	//       axis plane. Instead, we use the midpoint between max and min which
	//       will lie in the correct node IF the ray only intersects one node.
	//       Otherwise, it will still lie in an intersected node, so there are
	//       no false positives from this.
	uint8 intersectionMask = 0;
	{
		const float3 pointOnRay = (0.5f * (tmin + tmax)) * rayDirection;

		uint8 const firstChild =
			((pointOnRay.x >= centerRelativeToRay.x) ? 4 : 0) +
			((pointOnRay.y >= centerRelativeToRay.y) ? 2 : 0) +
			((pointOnRay.z >= centerRelativeToRay.z) ? 1 : 0);

		intersectionMask |= (1u << firstChild);
	}

	// We now check the points where the ray intersects the X, Y and Z plane.
	// If the intersection is within (ray_tmin, ray_tmax) then the intersection
	// point implies that two voxels will be touched by the ray. We find out
	// which voxels to mask for an intersection point at +X, +Y by setting
	// ALL voxels at +X and ALL voxels at +Y and ANDing these two masks.
	//
	// NOTE: When the intersection point is close enough to another axis plane,
	//       we must check both sides or we will get robustness issues.
	const float epsilon = 1e-4f;

	if (tmin <= tmid.x && tmid.x <= tmax)
	{
		const float3 pointOnRay = tmid.x * rayDirection;

		uint8 A = 0;
		if (pointOnRay.y >= centerRelativeToRay.y - epsilon) A |= 0xCC;
		if (pointOnRay.y <= centerRelativeToRay.y + epsilon) A |= 0x33;

		uint8 B = 0;
		if (pointOnRay.z >= centerRelativeToRay.z - epsilon) B |= 0xAA;
		if (pointOnRay.z <= centerRelativeToRay.z + epsilon) B |= 0x55;

		intersectionMask |= A & B;
	}
	if (tmin <= tmid.y && tmid.y <= tmax)
	{
		const float3 pointOnRay = tmid.y * rayDirection;

		uint8 C = 0;
		if (pointOnRay.x >= centerRelativeToRay.x - epsilon) C |= 0xF0;
		if (pointOnRay.x <= centerRelativeToRay.x + epsilon) C |= 0x0F;

		uint8 D = 0;
		if (pointOnRay.z >= centerRelativeToRay.z - epsilon) D |= 0xAA;
		if (pointOnRay.z <= centerRelativeToRay.z + epsilon) D |= 0x55;

		intersectionMask |= C & D;
	}
	if (tmin <= tmid.z && tmid.z <= tmax)
	{
		const float3 pointOnRay = tmid.z * rayDirection;

		uint8 E = 0;
		if (pointOnRay.x >= centerRelativeToRay.x - epsilon) E |= 0xF0;
		if (pointOnRay.x <= centerRelativeToRay.x + epsilon) E |= 0x0F;


		uint8 F = 0;
		if (pointOnRay.y >= centerRelativeToRay.y - epsilon) F |= 0xCC;
		if (pointOnRay.y <= centerRelativeToRay.y + epsilon) F |= 0x33;

		intersectionMask |= E & F;
	}

	return intersectionMask;
}

struct StackEntry
{
	uint32 index;
	uint8 childMask;
	uint8 visitMask;
};

__global__ void trace_paths(const FDAGTracer::TracePathsParams traceParams, const FDAG dag, const FColors colors, cudaSurfaceObject_t surface)
{
	// Target pixel coordinate
	const uint2 pixel = make_uint2(
		blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (pixel.x >= traceParams.width || pixel.y >= traceParams.height)
		return; // outside.

	uint32 DebugIndex = 0;
	Leaf cachedLeaf{}; // needed to iterate on the last few levels
	FPath path(0, 0, 0);

	{
		// Pre-calculate per-pixel data
		const float3 rayOrigin = make_float3(traceParams.cameraPosition);
		const float3 rayDirection = make_float3(normalize(traceParams.rayMin + pixel.x * traceParams.rayDDx + pixel.y * traceParams.rayDDy - traceParams.cameraPosition));

		const float3 rayDirectionInverse = make_float3(make_double3(1. / rayDirection.x, 1. / rayDirection.y, 1. / rayDirection.z));
		const uint8 rayChildOrder =
			(rayDirection.x < 0.f ? 4 : 0) +
			(rayDirection.y < 0.f ? 2 : 0) +
			(rayDirection.z < 0.f ? 1 : 0);

		// State
		uint32 level = 0;

		StackEntry stack[LEVELS];
		StackEntry cache;

		cache.index = 0;
		cache.childMask = Utils::ChildMask(dag.get_node(0, cache.index));
		cache.visitMask = cache.childMask & compute_intersection_mask<true>(0, path, dag, rayOrigin, rayDirection, rayDirectionInverse);

		// Traverse DAG
		for (;;)
		{
			// Ascend if there are no children left.
			{
				uint32 newLevel = level;
				while (newLevel > 0 && !cache.visitMask)
				{
					newLevel--;
					cache = stack[newLevel];
				}

				if (newLevel == 0 && !cache.visitMask)
				{
					path = FPath(0, 0, 0);
					break;
				}

				path.ascend(level - newLevel);
				level = newLevel;
			}

			// Find next child in order by the current ray's direction
			const uint8 nextChild = next_child(rayChildOrder, cache.visitMask);

			// Mark it as handled
			cache.visitMask &= ~(1u << nextChild);

			// Intersect that child with the ray
			{
				path.descend(nextChild);
				stack[level] = cache;
				level++;

				if (LEVELS - 2 - level == traceParams.DebugLevel)
				{
					DebugIndex = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
				}

				// If we're at the final level, we have intersected a single voxel.
				if (level == dag.levels)
				{
					break;
				}

				// Are we in an internal node?
				if (level < dag.leaf_level())
				{
					cache.index = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
					cache.childMask = Utils::ChildMask(dag.get_node(level, cache.index));
					cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
				}
				else
				{
					/* The second-to-last and last levels are different: the data
					 * of these two levels (2^3 voxels) are packed densely into a
					 * single 64-bit word.
					 */
					uint8 childMask;

					if (level == dag.leaf_level())
					{
						const uint32 addr = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
						cachedLeaf = dag.get_leaf(addr);
						childMask = cachedLeaf.get_first_child_mask();
					}
					else
					{
						childMask = cachedLeaf.get_second_child_mask(nextChild);
					}

					// No need to set the index for bottom nodes
					cache.childMask = childMask;
					cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
				}
			}
		}
	}
	
#if ENABLE_COLORS
	uint64 nof_leaves = 0;
	{
		uint32 level = 0;
		uint32 nodeIndex = 0;
		while (level < dag.leaf_level())
		{
			level++;

			// Find the current childmask and which subnode we are in
			const uint32 node = dag.get_node(level - 1, nodeIndex);
			const uint8 childMask = Utils::ChildMask(node);
			const uint8 child = path.child_index(level, dag.levels);

			// Make sure the node actually exists
			if (!(childMask & (1 << child)))
			{
				nof_leaves = uint64(-1);
				break;
			}

			//////////////////////////////////////////////////////////////////////////
			// Find out how many leafs are in the children preceding this
			//////////////////////////////////////////////////////////////////////////
			// If at final level, just count nof children preceding and exit
			if (level == dag.leaf_level())
			{
				for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild)
				{
					if (childMask & (1u << childBeforeChild))
					{
						const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
						const Leaf leaf = dag.get_leaf(childIndex);
						nof_leaves += Utils::Popcll(leaf.to_64());
					}
				}
				const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
				const Leaf leaf = dag.get_leaf(childIndex);
				const uint8 leafBitIndex =
					(((path.path.x & 0x1) == 0) ? 0 : 4) |
					(((path.path.y & 0x1) == 0) ? 0 : 2) |
					(((path.path.z & 0x1) == 0) ? 0 : 1) |
					(((path.path.x & 0x2) == 0) ? 0 : 32) |
					(((path.path.y & 0x2) == 0) ? 0 : 16) |
					(((path.path.z & 0x2) == 0) ? 0 : 8);
				nof_leaves += Utils::Popcll(leaf.to_64() & ((uint64(1) << leafBitIndex) - 1));

				break;
			}
			else
			{
				// Otherwise, fetch the next node (and accumulate leaves we pass by)
				for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild)
				{
					if (childMask & (1u << childBeforeChild))
					{
						const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
						const uint32 childNode = dag.get_node(level, childIndex);
						nof_leaves += colors.get_leaves_count(level, childNode);
					}
				}
				nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
			}
		}
	}
#endif
	
	const auto p = path.path;

	uint32 color = 0;
#if ENABLE_COLORS
	if (traceParams.DebugLevel == 0xFFFFFFFC)
	{
		color = nof_leaves;
	}
	if (traceParams.DebugLevel == 0xFFFFFFFD)
	{
		color = nof_leaves == uint64(-1) ? 0 : colors.get_color(nof_leaves);
	}
#endif
	if (traceParams.DebugLevel == 0xFFFFFFFE)
	{
		color = Utils::MurmurHash64(cachedLeaf.to_64());
	}
	if (traceParams.DebugLevel == 0xFFFFFFFF)
	{
		color = float3_to_rgb888(make_float3(p.x & 0xFF, p.y & 0xFF, p.z & 0xFF) / make_float3(0xFF));
	}
	if (0 <= traceParams.DebugLevel && traceParams.DebugLevel < LEVELS)
	{
		color = Utils::MurmurHash32(DebugIndex);
	}
	surf2Dwrite(color, surface, pixel.x * sizeof(uint32), pixel.y);
}

void FDAGTracer::ResolvePaths(
	const TracePathsParams& Params,
	const TGpuArray<uint32>& Dag,
	const TGpuArray<uint32>& Colors,
	const TGpuArray<uint64>& EnclosedLeaves)
{
	CUDA_CHECK_ERROR();

	ColorsBuffer.MapSurface();
	
	const dim3 block_dim = dim3(4, 64);
	const dim3 grid_dim = dim3(Width / block_dim.x + 1, Height / block_dim.y + 1);
	
	trace_paths <<< grid_dim, block_dim >>> (Params, { Dag }, { Colors, EnclosedLeaves }, ColorsBuffer.CudaSurface);

	CUDA_CHECK_ERROR();
	
	ColorsBuffer.UnmapSurface();
}