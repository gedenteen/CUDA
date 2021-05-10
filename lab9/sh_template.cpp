#include <GL/glew.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>

void checkErrors(std::string desc);

GLuint genRenderProg() { 
	//создаем шейдер - код в виде строки комплириуем и компонуем
	//преобразования выполняются на хосте, шейдеры выполняются на устройстве 
	//возвращает айди программы
	GLuint progHandle = glCreateProgram(); //создать программу, получить дескриптор
	GLuint vp = glCreateShader(GL_VERTEX_SHADER); //создать вершинный шейдер
	GLuint fp = glCreateShader(GL_FRAGMENT_SHADER); //создать фрагментный шейдер
	
 	//строки ниже:
		//версия вершинного шейдера
		//переменные (pos, color) длины 3
		//out = переменная будет передана фрагментному шейдеру
		//
		//void main()
			//gl_Poition - это встроенная переменная; расширяем позицию до 4-мерного 
			//4 координата для сдвижения в пространстве
			//инициализировать цвет 
	const char *vpSrc[] = {
		"#version 430\n", 
		"layout(location = 0) in vec3 pos;\
		layout(location = 1) in vec3 color;\
		out vec4 vs_color;\
		uniform mat4 MVP;\
		void main() {\
			gl_Position = MVP*vec4(pos,1);\
			vs_color=vec4(color,1.0);\
		}"
	};
	const char *fpSrc[] = {
		"#version 430\n",
		"in vec4 vs_color;\
		out vec4 fcolor;\
		void main() {\
			fcolor = vs_color;\
		}"
	};
	
	glShaderSource( //связать дескриптор шейдера и описание
		vp, //дескриптор шейдера
		2, //сколько строк
		vpSrc, //массив строк
		NULL);
	glShaderSource(fp, 2, fpSrc, NULL);

	glCompileShader(vp); //комплируем шейдер
	int rvalue; //return value 
	glGetShaderiv(vp, GL_COMPILE_STATUS, &rvalue);
	
	if (!rvalue) { //обработка ошибки
		fprintf(stderr, "Error in compiling vp\n");
		exit(30);
	}
	glAttachShader(progHandle, vp); //включить шейдер в программу
	
	//все то же самое с фрагментным шейдером:
	glCompileShader(fp);
	glGetShaderiv(fp, GL_COMPILE_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in compiling fp\n");
		exit(31);
	}
	glAttachShader(progHandle, fp);

	glLinkProgram(progHandle); //компонуем программу
	glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in linking sp\n");
		exit(32);
	}
	checkErrors("Render shaders");
	
	return progHandle; //вернуть дескриптор программы 
}
