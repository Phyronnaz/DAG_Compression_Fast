#pragma once

#include "Core.h"
#include <vector>
#include "gmath/Vector3.h"

/**
* An aabb defined by the min and max extrema.
*/
class FAABB
{
public:
	Vector3 Min = { 0, 0, 0 };
	Vector3 Max = { 0, 0, 0 };

	Vector3 GetCentre() const
	{
		return (Min + Max) * 0.5f;
	}
	Vector3 GetHalfSize() const
	{
		return (Max - Min) * 0.5f;
	}
	double GetVolume() const
	{
		const Vector3 D = Max - Min;
		return D.X * D.Y * D.Z;
	}
	double GetArea() const
	{
		const Vector3 D = Max - Min;
		return D.X * D.Y * 2.0f + D.X * D.Z * 2.0f + D.Z * D.Y * 2.0f;
	}
};

inline FAABB Combine(const FAABB& A, const FAABB& B)
{
	return { Vector3::Min(A.Min, B.Min), Vector3::Max(A.Max, B.Max) };
}

inline FAABB Combine(const FAABB& a, const Vector3& pt)
{
	return { Vector3::Min(a.Min, pt), Vector3::Max(a.Max, pt) };
}

inline FAABB MakeInverseExtremeAABB()
{
	return { Vector3(FLT_MAX), Vector3(-FLT_MAX) };
}
inline FAABB MakeAABB(const Vector3& min, const Vector3& max)
{
	return { min, max };
}
inline FAABB MakeAABB(const Vector3& position, const float radius)
{
	return { position - radius, position + radius };
}
inline FAABB MakeAABB(const Vector3* Positions, int32 NumPositions)
{
	FAABB Result = MakeInverseExtremeAABB();
	for (int32 Index = 0; Index < NumPositions; Index++)
	{
		Result = Combine(Result, Positions[Index]);
	}
	return Result;
}
inline bool Overlaps(const FAABB& A, const FAABB& B)
{
	return A.Max.X > B.Min.X && A.Min.X < B.Max.X
		&& A.Max.Y > B.Min.Y && A.Min.Y < B.Max.Y
		&& A.Max.Z > B.Min.Z && A.Min.Z < B.Max.Z;

}

inline FAABB MakeSquareAABB(const FAABB& AABB)
{
	const Vector3 HalfSize = AABB.GetHalfSize();
	const Vector3 Centre = AABB.GetCentre();
	const Vector3 NewHalfSize{ Vector3::Max(HalfSize.X, Vector3::Max(HalfSize.Y, HalfSize.Z)) };
	return { Centre - NewHalfSize, Centre + NewHalfSize };
}

// Split in morton order
inline std::vector<FAABB> SplitAABB(const std::vector<FAABB>& Parent)
{
	std::vector<FAABB> Return;
	Return.reserve(Parent.size() * 8);

	for (const auto& AABB : Parent) 
	{
		const auto Centre = AABB.GetCentre();
		const auto HalfSize = AABB.GetHalfSize();

		for (int32 Index = 0; Index < 8; Index++) 
		{
			auto NewCentre = Centre;
			NewCentre.X += HalfSize.X * (Index & 4 ? 0.5f : -0.5f);
			NewCentre.Y += HalfSize.Y * (Index & 2 ? 0.5f : -0.5f);
			NewCentre.Z += HalfSize.Z * (Index & 1 ? 0.5f : -0.5f);

			Return.push_back(MakeAABB(NewCentre - 0.5f * HalfSize, NewCentre + 0.5f * HalfSize));
		}
	}
	return Return;
}