#pragma once

#include<vector>

#define GLEW_STATIC
#include <gl\glew.h>

#include"vmath.h"

#define _USE_MATH_DEFINES	1
#include <math.h>

#define TORADIAN(a)  (a*((float)M_PI/180.0f))

#define YAW  -90.0f
#define PITCH  0.0f
#define SPEED  3.0f
#define SENSITIVITY  0.02f
#define ZOOM  45.0f
//#define DELTA 1.0f/60.0f
//#define DELTA 0.05f


enum ECameraMovement
{
	E_FORWARD,
	E_BACKARD,
	E_LEFT,
	E_RIGHT,
};

//const GLfloat YAW = -90.0f;
//const GLfloat PITCH = 0.0f;
//const GLfloat SPEED = 6.0f;
//const GLfloat SENSITIVITY = 0.25f;
//const GLfloat ZOOM = 45.0f;

class Camera
{

private:

	vmath::vec3 position;
	vmath::vec3 front;
	vmath::vec3 up;
	vmath::vec3 right;
	vmath::vec3 worldUp;

	GLfloat yaw;
	GLfloat pitch;

	GLfloat movementSpeed;
	GLfloat mouseSensitivity;
	GLfloat zoom;


	void UpdateCameraVectors()
	{
		
		vmath::vec3 front;
		front[0] = cosf(TORADIAN(this->yaw * cosf(TORADIAN(this->pitch)) )  );//x
		front[1] = sinf(TORADIAN(this->pitch));//y
		front[2] = sinf(TORADIAN(this->yaw * cosf(TORADIAN(this->pitch))));//z

		this->front = vmath::normalize(front);
		this->right = vmath::normalize(vmath::cross(this->front,this->worldUp));
		this->up = vmath::normalize(vmath::cross(this->right,this->front));


	}


public:
	Camera(vmath::vec3 v3Position = vmath::vec3(0.0f, 0.0f, 0.0f),
		vmath::vec3 v3UP = vmath::vec3(0.0f, 1.0f, 0.0f),
		GLfloat yaw = YAW, GLfloat pitch = PITCH) :front(vmath::vec3(0.0f, 0.0f, -1.0f)), movementSpeed(SPEED), mouseSensitivity(SENSITIVITY), zoom(ZOOM)
	{
		this->position = v3Position;
		this->worldUp = v3UP;
		this->yaw = yaw;
		this->pitch = pitch;
		this->UpdateCameraVectors();
	}

	Camera(GLfloat posX, GLfloat posY, GLfloat posZ, GLfloat upX, GLfloat upY, GLfloat upZ, GLfloat yaw, GLfloat pitch) :front(vmath::vec3(0.0f, 0.0f, -1.0f)), movementSpeed(SPEED), mouseSensitivity(SENSITIVITY), zoom(ZOOM)
	{
		this->position = vmath::vec3(posX, posY, posZ);
		this->worldUp = vmath::vec3(upX, upY, upZ);
		this->yaw = yaw;
		this->pitch = pitch;
		this->UpdateCameraVectors();
	}
	

	vmath::mat4 GetViewMatrix()
	{
		return vmath::lookat(this->position, this->position+ this->front, this->up);
	}

	void ProcessKeyBoard(ECameraMovement direction,GLfloat deltaTime)
	{
		GLfloat velocity = this->movementSpeed * deltaTime;
		

		switch (direction)
		{
		case E_FORWARD:
			this->position += velocity * this->front;
			break;
		case E_BACKARD:

			this->position -= velocity * this->front;

			break;
		case E_LEFT:

			this->position -= this->right * velocity;

			break;
		case E_RIGHT:
			this->position += this->right * velocity;
			break;
		default:
			break;
		}
	}

	void ProcessMouseMovements(GLfloat xOffset, GLfloat yOffset,GLboolean constarintPitch = true)
	{
		xOffset *= this->mouseSensitivity;
		yOffset *= this->mouseSensitivity;

		this->yaw = this->yaw + xOffset;
		this->pitch = this->pitch + yOffset;

		if (constarintPitch)
		{
			if (this->pitch > 360.0f)
			{
				this->pitch = 360.0f;
			}

			if (this->pitch < -360.0f)
			{
				this->pitch = -360.0f;
			}
		}

		this->UpdateCameraVectors();

	}

	void ProcessMouseScrool(GLfloat yOffset, GLfloat xOffset)
	{
		if (this->zoom >= 1.0f && this->zoom <= 45.0f)
		{
			this->zoom -= yOffset;
		}

		if (this->zoom <= 1.0f)
		{
			this->zoom = 1.0f;
		}

		if (this->zoom >= 45.0f)
		{
			this->zoom = 45.0f;
		}

	}

	GLfloat GetZoom()
	{
		return this->zoom;
	}

	
	//~Camera();

};

