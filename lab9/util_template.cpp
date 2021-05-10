#include <GL/glew.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtc/matrix_transform.hpp>

void checkErrors(std::string desc) { //||||||возможно это лучше перенести в sh_template.cpp
	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		fprintf(stderr, "OpenGL error in \"%s\": %s (%d)\n", desc.c_str(),
		gluErrorString(e), e);
		exit(20);
	}
}

const unsigned int window_width = 512;
const unsigned int window_height = 512;

GLuint bufferID;
GLuint progHandle;
GLuint genRenderProg();
const int num_of_verticies = 3;

int initBuffer() { //выделяем память, делаем ее текущей, инициализируем 
	glGenBuffers(1, &bufferID); //выделение памяти(кол-во буферов, массив идентификаторов)
	//буфер - это область памяти 
	glBindBuffer(GL_ARRAY_BUFFER, bufferID); //делаем буфер текущим
	static const GLfloat vertex_buffer_data[] = { //инициализируем массив
		-0.9f, -0.9f, -0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
		0.9f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f,
	};
	glBufferData( //выделить память и скопировать туда созданный массив
		GL_ARRAY_BUFFER, //тип памяти
		6*num_of_verticies*sizeof(float), //размер памяти
		vertex_buffer_data, //указатель на хосте
		GL_STATIC_DRAW); //только для чтения
	//glBindBuffer(GL_ARRAY_BUFFER, inA);
	return 0;
}

void camera() {
	glm::mat4 Projection = glm::perspective(glm::radians(60.0f),
		(float) window_width / (float)window_height, 0.1f, 0.0f);
	glm::mat4 View = glm::lookAt( //местонахождение камеры
		glm::vec3(1,1,2), // Камера находится в точке (x,y,z)
		glm::vec3(0,0,0), // и направлена на начало координат.
		glm::vec3(0,1,0) // Ось Y направлена вверх, ( 0,-1,0) - вниз.
	);
	
	glm::mat4 Model = glm::mat4(1.0f); //единичная матрица
	glm::mat4 mvp = Projection * View * Model; //матрица конечная - как и откуда смотрим
	
	GLuint MatrixID = glGetUniformLocation(progHandle, "MVP"); //передать матрицу шейдеру
	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]); //передаем только для чтения
}

void display() { //здесь выводится изображение
	progHandle = genRenderProg(); //см. sh_template.cpp
	glUseProgram(progHandle); //делаем программу текущей
	camera();
	GLint posPtr = glGetAttribLocation(progHandle, "pos"); //получить указатель на программу
	//pos связывается с буфером (массивом чисел, где координаты и цвета)
		//по имени "pos"
	glVertexAttribPointer( //атрибуты 
		posPtr, //переменная-вектор
		3, //3 значения пропускать для каждого элемента в массиве из initBuffer() 
		GL_FLOAT, //тип = вещественные числа 
		GL_FALSE, 
		24, //размерность масива
		0); //0 = с самого начала буфера
	glEnableVertexAttribArray(posPtr); //делаем указатель активным
	
	GLint colorPtr = glGetAttribLocation(progHandle, "color"); //цвет
	glVertexAttribPointer( //передать цвет
		colorPtr, //цвет
		3, //по 3 элемента передаем данные 
		GL_FLOAT, //вещественны числа
		GL_FALSE, 
		24, //размер в байтах
		(const GLvoid*)12); //смещение
	glEnableVertexAttribArray(colorPtr);
	glDrawArrays( //запустить программу
		GL_TRIANGLES, //интерпретировать данные как вершины треугольника
		0, //начинать с нуля байт
		num_of_verticies); //количество вершин
	glDisableVertexAttribArray(posPtr); //сделать неактивным указатель
	glDisableVertexAttribArray(colorPtr);
	//"Vertex" в функциях обозначает вершинные шейдеры
}

void myCleanup() { //освобождение ресурсов - сделать неактивным
	glDeleteBuffers(1, &bufferID);
	glDeleteProgram(progHandle);
}

