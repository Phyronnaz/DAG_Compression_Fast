#pragma once

#include "Core.h"
#include <glm/glm.hpp>

/**
* An aabb defined by the min and max extrema.
*/
class FAABB
{
public:
	glm::vec3 Min = { 0, 0, 0 };
	glm::vec3 Max = { 0, 0, 0 };

	glm::vec3 GetCentre() const
	{
		return (Min + Max) * 0.5f;
	}
	glm::vec3 GetHalfSize() const
	{
		return (Max - Min) * 0.5f;
	}
	float GetVolume() const
	{
		const glm::vec3 D = Max - Min;
		return D.x * D.y * D.z;
	}
	float GetArea() const
	{
		const glm::vec3 D = Max - Min;
		return D.x * D.y * 2.0f + D.x * D.z * 2.0f + D.z * D.y * 2.0f;
	}
};

inline FAABB Combine(const FAABB& A, const FAABB& B)
{
	return { min(A.Min, B.Min), max(A.Max, B.Max) };
}

inline FAABB Combine(const FAABB& a, const glm::vec3& pt)
{
	return { min(a.Min, pt), max(a.Max, pt) };
}

inline FAABB MakeInverseExtremeAABB()
{
	return { glm::vec3(FLT_MAX), glm::vec3(-FLT_MAX) };
}
inline FAABB MakeAABB(const glm::vec3& min, const glm::vec3& max)
{
	return { min, max };
}
inline FAABB MakeAABB(const glm::vec3& position, const float radius)
{
	return { position - radius, position + radius };
}
inline FAABB MakeAABB(const glm::vec3* Positions, int32 NumPositions)
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
	return A.Max.x > B.Min.x && A.Min.x < B.Max.x
		&& A.Max.y > B.Min.y && A.Min.y < B.Max.y
		&& A.Max.z > B.Min.z && A.Min.z < B.Max.z;

}

inline FAABB MakeSquareAABB(const FAABB& AABB)
{
	const glm::vec3 HalfSize = AABB.GetHalfSize();
	const glm::vec3 Centre = AABB.GetCentre();
	const glm::vec3 NewHalfSize{ glm::max(HalfSize.x, glm::max(HalfSize.y, HalfSize.z)) };
	return { Centre - NewHalfSize, Centre + NewHalfSize };
}