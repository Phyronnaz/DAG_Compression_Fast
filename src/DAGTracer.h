#pragma once

#include "Core.h"
#include "CudaGLBuffer.h"
#include "CudaMath.h"
#include "Array.h"

class FDAGTracer
{
public:
	const uint32 Width;
	const uint32 Height;	

	FDAGTracer(uint32 Width, uint32 Height);
	~FDAGTracer();

	struct TracePathsParams
	{
		uint32 width;
		uint32 height;

		double3 cameraPosition;
		double3 rayMin;
		double3 rayDDx;
		double3 rayDDy;
	};
	void ResolvePaths(const TracePathsParams& Params, const TStaticArray<uint32, EMemoryType::GPU>& Dag);

	template<typename T1, typename T2>
	inline static TracePathsParams GetTraceParams(
		const T1& View, 
		const uint32 Width, 
		const uint32 Height,
		const T2& AABB)
	{
		const auto Vector3ToDouble3 = [](auto v) { return make_double3(v.x, v.y, v.z); };
		
		const double3 Position  = Vector3ToDouble3(View.Position);
		const double3 Direction = Vector3ToDouble3(View.Rotation[2]);
		const double3 Up        = Vector3ToDouble3(View.Rotation[1]);
		const double3 Right     = Vector3ToDouble3(View.Rotation[0]);

		const double3 BoundsMin = Vector3ToDouble3(AABB.Min);
		const double3 BoundsMax = Vector3ToDouble3(AABB.Max);
		
		const double Fov = View.Fov / 2.0 * (3.14159265358979323846 / 180.);
		const double AspectRatio = double(Width) / double(Height);
		
		const double3 X = Right     * sin(Fov) * AspectRatio;
		const double3 Y = Up        * sin(Fov);
		const double3 Z = Direction * cos(Fov);

		const double3 BottomLeft  = Position + Z - Y - X;
		const double3 BottomRight = Position + Z - Y + X;
		const double3 TopLeft     = Position + Z + Y - X;

		const double3 Translation = -BoundsMin;
		const double3 Scale = make_double3(double(1 << LEVELS)) / (BoundsMax - BoundsMin);

		const double3 FinalPosition    = (Position    + Translation) * Scale;
		const double3 FinalBottomLeft  = (BottomLeft  + Translation) * Scale;
		const double3 FinalTopLeft     = (TopLeft     + Translation) * Scale;
		const double3 FinalBottomRight = (BottomRight + Translation) * Scale;
		const double3 Dx = (FinalBottomRight - FinalBottomLeft) * (1.0 / Width);
		const double3 Dy = (FinalTopLeft     - FinalBottomLeft) * (1.0 / Height);

		TracePathsParams params;

		params.width = Width;
		params.height = Height;

		params.cameraPosition = FinalPosition;
		params.rayMin = FinalBottomLeft;
		params.rayDDx = Dx;
		params.rayDDy = Dy;

		return params;
	}
	
	inline GLuint GetColorsImage() const
	{
		return ColorsImage;
	}

private:
	uint8* NextChildLookupTable = nullptr;
	FCudaGLBuffer ColorsBuffer;
	GLuint ColorsImage = 0;
};