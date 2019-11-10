#pragma once

#include "Core.h"
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include "glm_extensions.h"

struct FOrientation
{
public:
	glm::mat3 Rotation = glm::mat3(1.0f);
	glm::vec3 Position = glm::vec3(0.0f);
	glm::vec3 Scale = glm::vec3(1.0f);

	FOrientation() = default;
	FOrientation(const glm::vec3& InPosition, const glm::vec3& InDirection, const glm::vec3& InUp)
	{
		LookAt(InPosition, InDirection, InUp);
		Scale = glm::vec3(1.0f);
	}

	void LookAt(const glm::vec3& Eye, const glm::vec3& Center, const glm::vec3& Up)
	{
		Position = Eye;
		const glm::vec3 Direction = normalize(Eye - Center);
		const glm::vec3 Right = normalize(cross(Up, normalize(Direction)));
		const glm::vec3 NewUp = normalize(cross(Direction, Right));
		Rotation = glm::mat3(Right, NewUp, Direction);
	}
	void LookAt(const glm::vec3& Eye, const glm::vec3& Center)
	{
		Position = Eye;
		const glm::vec3 Direction = normalize(Eye - Center);
		const glm::vec3 Right = normalize(perp(Direction));
		const glm::vec3 NewUp = normalize(cross(Direction, Right));
		Rotation = glm::mat3(Right, NewUp, Direction);
	}

	glm::mat4 GetViewMatrix() const 
	{
		const glm::mat4 InvRotation = glm::mat4(transpose(Rotation));
		return scale(glm::mat4(1.0f), 1.f / Scale) * InvRotation * translate(glm::mat4(1.0f), -Position);
	}
	glm::mat4 GetViewMatrixInverted() const
	{
		return translate(glm::mat4(1.0f), Position) * glm::mat4(Rotation) * scale(glm::mat4(1.0f), Scale);
	}

	void Roll(const float Angle)
	{
		Rotation = Rotation * glm::mat3(rotate(glm::mat4(1.0f), Angle, glm::vec3(0.0f, 0.0f, -1.0f)));
	}
	void Yaw(const float Angle)
	{
		Rotation = Rotation * glm::mat3(rotate(glm::mat4(1.0f), Angle, glm::vec3(0.0f, 1.0f, 0.0f)));
	}
	void Pitch(const float Angle)
	{
		Rotation = Rotation * glm::mat3(rotate(glm::mat4(1.0f), Angle, glm::vec3(1.0f, 0.0f, 0.0f)));
	}
};

struct FView : FOrientation
{
	float Fov = 60.0;
	float AspectRatio = 1.0;
	float Near = 1.0;
	float Far = 500.0;

	glm::mat4 GetProjectionMatrix() const
	{
		return glm::make_perspective(Fov, AspectRatio, Near, Far);
	}
	glm::mat4 GetProjectionMatrixInverted() const
	{
		return glm::make_perspective_inv(Fov, AspectRatio, Near, Far);
	}
};

struct FOrthographicView : FOrientation
{
	float Left = 0;
	float Right = 0;
	float Bottom = 0;
	float Top = 0;
	float Near = 1.0;
	float Far = 500.0;

	glm::mat4 GetProjectionMatrix() const
	{
		return glm::make_ortho(Left, Right, Bottom, Top, Near, Far);
	}
	glm::mat4 GetProjectionMatrixInverted() const
	{
		return glm::make_ortho_inv(Left, Right, Bottom, Top, Near, Far);
	}
};

template<typename T>
inline glm::mat4 GetProjectionViewMatrix(const T& View)
{
	return View.GetProjectionMatrix() * View.GetViewMatrix();
}
template<typename T>
inline glm::mat4 GetProjectionViewMatrixInverted(const T& View)
{
	return View.GetViewMatrixInverted() * View.GetProjectionMatrixInverted();
}