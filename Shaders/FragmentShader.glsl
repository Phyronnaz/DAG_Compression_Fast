#extension GL_ARB_shader_atomic_counters : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_ARB_gpu_shader_int64 : enable

uint32_t splitBy3_32(uint32_t x)
{
	x = x & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x | (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x | (x << 8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x | (x << 4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x | (x << 2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
};
uint32_t mortonEncode32(uint32_t x, uint32_t y, uint32_t z)
{
	return splitBy3_32(x) << 2 | splitBy3_32(y) << 1 | splitBy3_32(z);
};

uint64_t splitBy3_64(uint32_t a)
{
	uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
	x = (x | x << 32) & 0x1f00000000fffful; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = (x | x << 16) & 0x1f0000ff0000fful; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = (x | x << 8) & 0x100f00f00f00f00ful; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = (x | x << 4) & 0x10c30c30c30c30c3ul; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = (x | x << 2) & 0x1249249249249249ul;
	return x;
};
uint64_t mortonEncode64(uint32_t x, uint32_t y, uint32_t z)
{
	return splitBy3_64(x) << 2 | splitBy3_64(y) << 1 | splitBy3_64(z);
};

vec4 SRGBtoLINEAR(vec4 srgbIn)
{
	vec3 bLess = step(vec3(0.04045), srgbIn.xyz);
	vec3 linOut = mix(srgbIn.xyz / vec3(12.92), pow((srgbIn.xyz + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
	return vec4(linOut, srgbIn.w);
}

vec4 LINEARtoSRGB(vec4 linIn)
{
	vec3 bLess = step(vec3(0.0031308), linIn.xyz);
	vec3 srgbOut = mix(linIn.xyz * vec3(12.92), vec3(1.055) * pow(linIn.xyz, vec3(1.0 / 2.4)) - vec3(0.055), bLess);
	return vec4(srgbOut, linIn.w);
}

void swap(inout int a[5], int i, int j)
{
	int tmp = a[i];
	a[i] = a[j];
	a[j] = tmp;
}

void sort(inout int a[5])
{
	// Sort first two pairs
	if (a[1] < a[0]) swap(a, 0, 1);
	if (a[3] < a[2]) swap(a, 2, 3);
	// Sort pairs by larger element
	if (a[3] < a[1])
	{
		swap(a, 0, 2);
		swap(a, 1, 3);
	}
	// A = [a,b,c,d,e] with a < b < d and c < d
	// insert e into [a,b,d]
	int b[4];
	if (a[4] < a[1])
	{
		if (a[4] < a[0])
		{
			b[0] = a[4];
			b[1] = a[0];
			b[2] = a[1];
			b[3] = a[3];
		}
		else
		{
			b[0] = a[0];
			b[1] = a[4];
			b[2] = a[1];
			b[3] = a[3];
		}
	}
	else
	{
		if (a[4] < a[3])
		{
			b[0] = a[0];
			b[1] = a[1];
			b[2] = a[4];
			b[3] = a[3];
		}
		else
		{
			b[0] = a[0];
			b[1] = a[1];
			b[2] = a[3];
			b[3] = a[4];
		}
	}
	// insert c into the first three elements of B
	if (a[2] < b[1])
	{
		if (a[2] < b[0])
		{
			a[0] = a[2];
			a[1] = b[0];
			a[2] = b[1];
			a[3] = b[2];
			a[4] = b[3];
		}
		else
		{
			a[1] = a[2];
			a[0] = b[0];
			a[2] = b[1];
			a[3] = b[2];
			a[4] = b[3];
		}
	}
	else
	{
		if (a[2] < b[2])
		{
			a[2] = a[2];
			a[0] = b[0];
			a[1] = b[1];
			a[3] = b[2];
			a[4] = b[3];
		}
		else
		{
			a[3] = a[2];
			a[0] = b[0];
			a[1] = b[1];
			a[2] = b[2];
			a[4] = b[3];
		}
	}
}

flat in int axis_id;
in vec2 uv;
in vec3 fs_position0;
in vec3 fs_position1;
in vec3 fs_position2;
in vec3 fs_normal;
in vec3 barycoords;
flat in vec3 fs_gnormal;

#define JOIN_IMPL(A, B) A ## B
#define JOIN(A, B) JOIN_IMPL(A, B)
#define MORTON_TYPE JOIN(uint, JOIN(MORTON_BITS, _t))

layout(binding = 0) uniform atomic_uint frag_count;
layout(binding = 0, std430) restrict coherent buffer item_buffer_block0 { MORTON_TYPE position_ssbo[]; };

uniform int grid_dim;
layout(binding = 0) uniform sampler2D u_BaseColorSampler;
layout(binding = 1) uniform sampler2D u_NormalSampler;

void main()
{
	///////////////////////////////////////////////////////////////////////
	// Fetch color (once per shader invocation)
	///////////////////////////////////////////////////////////////////////
	vec4 base_color = texture2D(u_BaseColorSampler, uv);

	vec3 normal = texture(u_NormalSampler, uv).rgb;

	if (base_color.a < 0.1f) return;

	vec3 subvoxel_pos = vec3(
		gl_FragCoord.x,
		gl_FragCoord.y,
		gl_FragCoord.z * grid_dim);

	// Conservative in z
	float dzdx = 0.5 * dFdxFine(gl_FragCoord.z) * grid_dim;
	float dzdy = 0.5 * dFdyFine(gl_FragCoord.z) * grid_dim;
	int apa[5] = {
		clamp(int(subvoxel_pos.z), 0, grid_dim - 1),
		clamp(int(subvoxel_pos.z + dzdx + dzdy), 0, grid_dim - 1),
		clamp(int(subvoxel_pos.z + dzdx - dzdy), 0, grid_dim - 1),
		clamp(int(subvoxel_pos.z - dzdx + dzdy), 0, grid_dim - 1),
		clamp(int(subvoxel_pos.z - dzdx - dzdy), 0, grid_dim - 1)
	};
	sort(apa);
	for (int i = 0; i < 1; ++i)
	{
		if (i == 0 || apa[i] != apa[i - 1])
		{
			uvec3 subvoxel_coord2 = uvec3(clamp(uvec2(subvoxel_pos.xy), uvec2(0), uvec2(grid_dim - 1)), uint(apa[i]));
			if (axis_id == 1)
			{
				subvoxel_coord2.xyz = subvoxel_coord2.zyx;
			}
			else if (axis_id == 2)
			{
				subvoxel_coord2.xyz = subvoxel_coord2.xzy;
			}
			else if (axis_id == 3)
			{
				subvoxel_coord2.xyz = subvoxel_coord2.yxz;
			}
			uint32_t idx = atomicCounterIncrement(frag_count);
			position_ssbo[idx] = JOIN(mortonEncode, MORTON_BITS)(subvoxel_coord2.x, subvoxel_coord2.y, subvoxel_coord2.z);
		}
	}
}