#include "DAGTracer.h"

#pragma warning (disable: 4244)

#define IN_ORDER_TRAVERSAL 1

///////////////////////////////////////////////////////////////////////////////
// Having a different child order than %000 means that we don't  process the 
// children in order 0,2,3,4,5,6,7. For example, child_order == %001 means that 
// we should do high z-values before low z values. 
// It all boils down to that the order is obtained as 
// child_number[child_order] = child_number[%000] ^ child_order
///////////////////////////////////////////////////////////////////////////////
HOST_DEVICE uint8 GetNextChild(uint8_t ChildOrder, uint8_t TestMask, uint8_t ChildMask)
{
	for (int32 Index = 0; Index < 8; Index++) 
	{
		const uint8 ChildInOrder = Index ^ ChildOrder;
		const bool ChildExists = (ChildMask & (1 << ChildInOrder)) != 0;
		const bool ChildIsTaken = (~TestMask & (1 << ChildInOrder)) != 0;
		if (ChildExists && !ChildIsTaken) return ChildInOrder;
	}
	return 0;
};

///////////////////////////////////////////////////////////////////////////////
// In the lookup table, we want to maintain locality as much as we can. 
// The "child_order" is very likely to be the same (at least for primary rays)
// between pixels. The "testmask" is probably more volatile than the 
// "childmask", so do that last.
///////////////////////////////////////////////////////////////////////////////
uint8* CalculateNextChildLookupTable()
{
	uint8* LUT = new uint8[8 * 256 * 256];
	for (uint32 ChildOrder = 0; ChildOrder < 8; ChildOrder++) 
	{
		for (uint32 ChildMask = 0; ChildMask < 256; ChildMask++) 
		{
			for (uint32 TestMask = 0; TestMask < 256; TestMask++) 
			{
				LUT[ChildOrder * (256 * 256) + ChildMask * 256 + TestMask] = GetNextChild(ChildOrder, TestMask, ChildMask);
			}
		}
	}
	return LUT; 
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

FDAGTracer::FDAGTracer(uint32 Width, uint32 Height)
	: Width(Width)
	, Height(Height)
{
	uint8* NextChildLookupTable_CPU = CalculateNextChildLookupTable();
	constexpr int32 LookupTableSize = 8 * 256 * 256 * sizeof(uint8); 
	cudaMalloc(&NextChildLookupTable, LookupTableSize);
	cudaMemcpy(NextChildLookupTable, NextChildLookupTable_CPU, LookupTableSize, cudaMemcpyHostToDevice);
	delete[] NextChildLookupTable;
	
	glGenTextures(1, &ColorsImage);
	glBindTexture(GL_TEXTURE_2D, ColorsImage);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int32)Width, (int32)Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
	ColorsBuffer.RegisterResource(ColorsImage);
}

FDAGTracer::~FDAGTracer()
{
	ColorsBuffer.UnregisterResource();
	glDeleteTextures(1, &ColorsImage);
}

///////////////////////////////////////////////////////////////////////////////
// Function for intersecting a node, returning a bitmask with the 
// childnodes that are intersected (ignoring whether they exist).
///////////////////////////////////////////////////////////////////////////////
inline __device__ uint8_t getIntersectionMask(const uint3 & stack_path, const uint32_t nof_levels,
												const uint32_t stack_level, const float3 & ray_o,
												const float3 & ray_d, const float3 &inv_ray_dir)
{
	///////////////////////////////////////////////////////////////////////////
	// Node center is 0.5 * (aabb_min + aabb_max), which reduces to:
	// NOTE: This is probably of by 0.5 right now...
	///////////////////////////////////////////////////////////////////////////
	uint32_t shift = nof_levels - (stack_level);
	uint32_t node_radius = 1 << (shift - 1);
	float3 node_center = make_float3(stack_path << shift) + make_float3(node_radius);

	///////////////////////////////////////////////////////////////////////////
	// Find the t-values at which the ray intersects the axis-planes of the 
	// node: ray_o + tmid * ray_d = node_center   (component wise)
	///////////////////////////////////////////////////////////////////////////
	float3 tmid = (node_center - ray_o) * inv_ray_dir;

	///////////////////////////////////////////////////////////////////////////
	// Now find the t values at which the ray intersects the parent node,
	// by calculating the t-range for traveling through a "slab" from the 
	// center to each side of the node. 
	///////////////////////////////////////////////////////////////////////////
	float ray_tmin, ray_tmax;
	{
		float3 slab_radius = node_radius * abs(inv_ray_dir);
		float3 tmin = tmid - slab_radius;
		ray_tmin = fmax(max(tmin), 0.0f);
		float3 tmax = tmid + slab_radius;
		ray_tmax = min(tmax);
	}

	///////////////////////////////////////////////////////////////////////////
	// Find the first child that is intersected. 
	// NOTE: We assume that we WILL hit one child, since we assume that the 
	//       parents bounding box is hit.
	// NOTE: To safely get the correct node, we cannot use o+ray_tmin*d as the
	//       intersection point, since this point might lie too close to an 
	//       axis plane. Instead, we use the midpoint between max and min which
	//       will lie in the correct node IF the ray only intersects one node. 
	//       Otherwise, it will still lie in an intersected node, so there are 
	//       no false positives from this. 
	///////////////////////////////////////////////////////////////////////////
	uint8_t intersection_mask = 0;
	{
		uint8_t first_hit_child;
		float3 point_on_ray_segment = ray_o + 0.5f * (ray_tmin + ray_tmax) * ray_d;
		first_hit_child = point_on_ray_segment.x >= node_center.x ? 4 : 0;
		first_hit_child += point_on_ray_segment.y >= node_center.y ? 2 : 0;
		first_hit_child += point_on_ray_segment.z >= node_center.z ? 1 : 0;
		intersection_mask |= (1 << first_hit_child);
	}

	///////////////////////////////////////////////////////////////////////////
	// We now check the points where the ray intersects the X, Y and Z plane. 
	// If the intersection is within (ray_tmin, ray_tmax) then the intersection
	// point implies that two voxels will be touched by the ray. We find out 
	// which voxels to mask for an intersection point at +X, +Y by setting
	// ALL voxels at +X and ALL voxels at +Y and ANDing these two masks. 
	// 
	// NOTE: When the intersection point is close enough to another axis plane, 
	//       we must check both sides or we will get robustness issues. 
	///////////////////////////////////////////////////////////////////////////
	const float epsilon = 0.0001f;
	float3 pointOnRaySegment0 = ray_o + tmid.x*ray_d;
	uint32_t A = (abs(node_center.y - pointOnRaySegment0.y) < epsilon) ? (0xCC | 0x33) : ((pointOnRaySegment0.y >= node_center.y ? 0xCC : 0x33));
	uint32_t B = (abs(node_center.z - pointOnRaySegment0.z) < epsilon) ? (0xAA | 0x55) : ((pointOnRaySegment0.z >= node_center.z ? 0xAA : 0x55));
	intersection_mask |= (tmid.x < ray_tmin || tmid.x > ray_tmax) ? 0 : (A & B);
	float3 pointOnRaySegment1 = ray_o + tmid.y*ray_d;
	uint32_t C = (abs(node_center.x - pointOnRaySegment1.x) < epsilon) ? (0xF0 | 0x0F) : ((pointOnRaySegment1.x >= node_center.x ? 0xF0 : 0x0F));
	uint32_t D = (abs(node_center.z - pointOnRaySegment1.z) < epsilon) ? (0xAA | 0x55) : ((pointOnRaySegment1.z >= node_center.z ? 0xAA : 0x55));
	intersection_mask |= (tmid.y < ray_tmin || tmid.y > ray_tmax) ? 0 : (C & D);
	float3 pointOnRaySegment2 = ray_o + tmid.z*ray_d;
	uint32_t E = (abs(node_center.x - pointOnRaySegment2.x) < epsilon) ? (0xF0 | 0x0F) : ((pointOnRaySegment2.x >= node_center.x ? 0xF0 : 0x0F));
	uint32_t F = (abs(node_center.y - pointOnRaySegment2.y) < epsilon) ? (0xCC | 0x33) : ((pointOnRaySegment2.y >= node_center.y ? 0xCC : 0x33));
	intersection_mask |= (tmid.z < ray_tmin || tmid.z > ray_tmax) ? 0 : (E & F);

	///////////////////////////////////////////////////////////////////////////
	// Checking ray_tmin > ray_tmax is not generally safe (it happens sometimes
	// due to float precision). But we still use this test to stop rays that do 
	// not hit the root node. 
	///////////////////////////////////////////////////////////////////////////	
	return (stack_level == 0 && ray_tmin >= ray_tmax) ? 0 : intersection_mask;
};

HOST_DEVICE uint32 float3_to_rgb888(float3 c)
{
	float R = fmin(1.0f, fmax(0.0f, c.x));
	float G = fmin(1.0f, fmax(0.0f, c.y));
	float B = fmin(1.0f, fmax(0.0f, c.z));
	return	(uint32_t(R * 255.0f) << 0) |
		(uint32_t(G * 255.0f) << 8) |
		(uint32_t(B * 255.0f) << 16);
}

__global__ void
primary_rays_kernel(
	const FDAGTracer::TracePathsParams Params, 
	const TStaticArray<uint32, EMemoryType::GPU> Dag, 
	uint8_t* const d_next_child_lookup_table,
	cudaSurfaceObject_t surface)
{
	///////////////////////////////////////////////////////////////////////////
	// Screen coordinates, discard outside
	///////////////////////////////////////////////////////////////////////////
	uint2 coord = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (coord.x >= Params.width || coord.y >= Params.height) return; 

	///////////////////////////////////////////////////////////////////////////
	// Calculate ray for pixel
	///////////////////////////////////////////////////////////////////////////
	const float3 ray_o = make_float3(Params.cameraPosition); 
	const float3 ray_d = make_float3(normalize((Params.rayMin + coord.x * Params.rayDDx + coord.y * Params.rayDDy) - Params.cameraPosition)); 
	float3 inv_ray_dir = 1.0f / ray_d;

	///////////////////////////////////////////////////////////////////////////
	// Stack
	///////////////////////////////////////////////////////////////////////////
	struct StackEntry{
		uint32_t node_idx;
		uint32_t masks; 
	} stack[LEVELS];
	uint3		stack_path = make_uint3(0,0,0);
	int			stack_level = 0; 


	///////////////////////////////////////////////////////////////////////////
	// If intersecting the root node, push it on stack
	///////////////////////////////////////////////////////////////////////////
	stack[0].node_idx = 0; 
	stack[0].masks = Dag[0] & 0xFF;
	stack[0].masks |= (stack[0].masks & getIntersectionMask(stack_path, LEVELS, stack_level, ray_o, ray_d, inv_ray_dir)) << 8;

	///////////////////////////////////////////////////////////////////////////
	// Traverse until stack is empty (all root children are processed), or 
	// until a leaf node is intersected. 
	///////////////////////////////////////////////////////////////////////////
	uint64_t current_leafmask; 
	StackEntry curr_se = stack[0]; // Current stack entry

	while (stack_level >= 0)
	{
		///////////////////////////////////////////////////////////////////////
		// If no children left to test, roll back stack
		///////////////////////////////////////////////////////////////////////
		if ((curr_se.masks & 0xFF00) == 0x0) {
			stack_level = stack_level - 1;
			stack_path = make_uint3(stack_path.x >> 1, stack_path.y >> 1, stack_path.z >> 1);
			while ((stack[stack_level].masks & 0xFF00) == 0x0 && stack_level >= 0) {
				stack_level = stack_level - 1;
				stack_path = make_uint3(stack_path.x >> 1, stack_path.y >> 1, stack_path.z >> 1);
			}
			if (stack_level < 0) break;
			curr_se = stack[stack_level];
		}

		///////////////////////////////////////////////////////////////////////
		// Figure out which child to test next and create its path and 
		// bounding box
		///////////////////////////////////////////////////////////////////////
#if IN_ORDER_TRAVERSAL
		int child_order =	ray_d.x < 0 ? 4 : 0;
		child_order		+=	ray_d.y < 0 ? 2 : 0;
		child_order		+=	ray_d.z < 0 ? 1 : 0;
		uint8_t next_child = d_next_child_lookup_table[child_order * (256 * 256) + (curr_se.masks & 0xFF) * 256 + (curr_se.masks >> 8)];
#else
		uint8_t next_child = 31 - __clz(node_testmask);
#endif

		curr_se.masks &= ~(1 << (next_child + 8));

		///////////////////////////////////////////////////////////////////////
		// Check this child for intersection with ray
		///////////////////////////////////////////////////////////////////////
		{
			uint32_t node_offset = __popc((curr_se.masks & 0xFF) & ((1 << next_child) - 1)) + 1;

			stack_path = make_uint3(
				(stack_path.x << 1) | ((next_child & 0x4) >> 2),
				(stack_path.y << 1) | ((next_child & 0x2) >> 1),
				(stack_path.z << 1) | ((next_child & 0x1) >> 0));
			stack_level += 1;

			constexpr int32 nof_levels = LEVELS - 2;

			///////////////////////////////////////////////////////////////////
			// If we are at the final level where we have intersected a 1x1x1
			// voxel. We are done. 
			///////////////////////////////////////////////////////////////////
			if (stack_level == nof_levels) {
				break;
			}
			
			///////////////////////////////////////////////////////////////////
			// As I have intersected (and I am not finished) the current 
			// stack entry must be saved to the stack
			///////////////////////////////////////////////////////////////////
			stack[stack_level - 1] = curr_se;
			
			///////////////////////////////////////////////////////////////////
			// Now let's see if we are at the 2x2x2 level, in which case we
			// need to pick a child-mask from previously fetched leaf-masks
			// (node_idx does not matter until we are back)
			///////////////////////////////////////////////////////////////////
			if (stack_level == nof_levels - 1) {
				curr_se.masks = (current_leafmask >> next_child * 8) & 0xFF;
			}
			///////////////////////////////////////////////////////////////////
			// If the child level is the "leaf" level (containing 4x4x4 
			// leafmasks), create a fake node for the next pass
			///////////////////////////////////////////////////////////////////
			else if (stack_level == nof_levels - 2) {
				uint32_t leafmask_address = Dag[curr_se.node_idx + node_offset]; 
				///////////////////////////////////////////////////////////////
				// Shouldn't there be a faster way to get the 8 bit mask from 
				// a 64 bit word... Without a bunch of compares? 
				///////////////////////////////////////////////////////////////
				uint32_t leafmask0 = Dag[leafmask_address];
				uint32_t leafmask1 = Dag[leafmask_address + 1];
				current_leafmask = uint64_t(leafmask1) << 32 | uint64_t(leafmask0); 
				curr_se.masks =
					((current_leafmask & 0x00000000000000FF) == 0 ? 0 : 1 << 0) |
					((current_leafmask & 0x000000000000FF00) == 0 ? 0 : 1 << 1) |
					((current_leafmask & 0x0000000000FF0000) == 0 ? 0 : 1 << 2) |
					((current_leafmask & 0x00000000FF000000) == 0 ? 0 : 1 << 3) |
					((current_leafmask & 0x000000FF00000000) == 0 ? 0 : 1 << 4) |
					((current_leafmask & 0x0000FF0000000000) == 0 ? 0 : 1 << 5) |
					((current_leafmask & 0x00FF000000000000) == 0 ? 0 : 1 << 6) |
					((current_leafmask & 0xFF00000000000000) == 0 ? 0 : 1 << 7);
			}
			///////////////////////////////////////////////////////////////
			// If we are at an internal node, push the child on the stack
			///////////////////////////////////////////////////////////////
			else {
				curr_se.node_idx = Dag[curr_se.node_idx + node_offset]; 
				curr_se.masks = Dag[curr_se.node_idx] & 0xFF;
			}

			curr_se.masks |= (curr_se.masks & getIntersectionMask(stack_path, nof_levels, stack_level, ray_o, ray_d, inv_ray_dir)) << 8;

		}
		///////////////////////////////////////////////////////////////////////
		// If it does not intersect, do nothing. Proceed at same level.
		///////////////////////////////////////////////////////////////////////
	}
	///////////////////////////////////////////////////////////////////////////
	// Done, stack_path is closest voxel (or 0 if none found)
	///////////////////////////////////////////////////////////////////////////

	const auto path = stack_path;
	
	const float3 color = make_float3(path.x & 0xFF, path.y & 0xFF, path.z & 0xFF) / make_float3(0xFF);
	surf2Dwrite(float3_to_rgb888(color), surface, coord.x  * sizeof(uint32), coord.y);
}

void FDAGTracer::ResolvePaths(const TracePathsParams& Params, const TStaticArray<uint32, EMemoryType::GPU>& Dag)
{
	ColorsBuffer.MapSurface();

	dim3 block_dim = dim3(8, 32);
	dim3 grid_dim = dim3(Width / block_dim.x + 1, Height / block_dim.y + 1);
	primary_rays_kernel <<<grid_dim, block_dim >>>(Params, Dag, NextChildLookupTable, ColorsBuffer.CudaSurface);

	ColorsBuffer.UnmapSurface();
}