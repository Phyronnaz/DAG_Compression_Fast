#pragma once

#include "Core.h"
#include "CudaGLBuffer.h"
#include "CudaMath.h"
#include "Array.h"
#include "Utils/Aabb.h"

#include "gmath/Vector3.h"
#include "gmath/Matrix3x3.h"

struct FCameraView
{
	static constexpr float Fov = 60.f;

	Vector3 Position = { 0.f, 0.f, 0.f };
	Matrix3x3 Rotation = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

	inline Vector3 Right() const
	{
		// No idea why minus, but else it's inverted
		return { -Rotation.D00, -Rotation.D01, -Rotation.D02 };
	}
	inline Vector3 Up() const
	{
		return { Rotation.D10, Rotation.D11, Rotation.D12 };
	}
	inline Vector3 Forward() const
	{
		return { Rotation.D20, Rotation.D21, Rotation.D22 };
	}
};

class FDAGTracer
{
public:
	const uint32 Width;
	const uint32 Height;	

	FDAGTracer(uint32 Width, uint32 Height);
	~FDAGTracer();

	struct TracePathsParams
	{
		// if 0xFFFFFFFF will use world position as color instead
		uint32 DebugLevel = 0;
		
		uint32 width;
		uint32 height;

		double3 cameraPosition;
		double3 rayMin;
		double3 rayDDx;
		double3 rayDDy;
	};
	void ResolvePaths(
		const TracePathsParams& Params, 
		const TGpuArray<uint32>& Dag, 
		const TGpuArray<uint32>& Colors, 
		const TGpuArray<uint64>& EnclosedLeaves);

	inline static TracePathsParams GetTraceParams(
		uint32 DebugLevel,
		const FCameraView& View, 
		const uint32 Width, 
		const uint32 Height,
		const FAABB& AABB)
	{
		const auto Vector3ToDouble3 = [](auto v) { return make_double3(v.X, v.Y, v.Z); };
		
		const double3 Position  = Vector3ToDouble3(View.Position);
		const double3 Direction = Vector3ToDouble3(View.Forward());
		const double3 Up        = Vector3ToDouble3(View.Up());
		const double3 Right     = Vector3ToDouble3(View.Right());

		const double3 BoundsMin = Vector3ToDouble3(AABB.Min);
		const double3 BoundsMax = Vector3ToDouble3(AABB.Max);
		
		const double Fov = View.Fov / 2.0 * (3.14159265358979323846 / 180.);
		const double AspectRatio = double(Width) / double(Height);
		
		const double3 X = Right     * std::sin(Fov) * AspectRatio;
		const double3 Y = Up        * std::sin(Fov);
		const double3 Z = Direction * std::cos(Fov);

		const double3 BottomLeft  = Position + Z - Y - X;
		const double3 BottomRight = Position + Z - Y + X;
		const double3 TopLeft     = Position + Z + Y - X;

		const double3 Translation = -BoundsMin;
		const double3 Scale = make_double3(double(1u << LEVELS)) / (BoundsMax - BoundsMin);

		const double3 FinalPosition    = (Position    + Translation) * Scale;
		const double3 FinalBottomLeft  = (BottomLeft  + Translation) * Scale;
		const double3 FinalTopLeft     = (TopLeft     + Translation) * Scale;
		const double3 FinalBottomRight = (BottomRight + Translation) * Scale;
		const double3 Dx = (FinalBottomRight - FinalBottomLeft) * (1.0 / Width);
		const double3 Dy = (FinalTopLeft     - FinalBottomLeft) * (1.0 / Height);

		TracePathsParams Params;

		Params.DebugLevel = DebugLevel;
		
		Params.width = Width;
		Params.height = Height;

		Params.cameraPosition = FinalPosition;
		Params.rayMin = FinalBottomLeft;
		Params.rayDDx = Dx;
		Params.rayDDy = Dy;

		return Params;
	}
	
	inline GLuint GetColorsImage() const
	{
		return ColorsImage;
	}

private:
	FCudaGLBuffer ColorsBuffer;
	GLuint ColorsImage = 0;
};