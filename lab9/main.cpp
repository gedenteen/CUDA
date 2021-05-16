#include "constants.h"

void initGL();
int initBuffer();
void display();
void myCleanup();

GLFWwindow* window;

int main(){
	initGL();
	initBuffer(); //функция из util_template.cpp
	do {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//glClearColor(0.7,0.7,0.7,1.0);
		glPointSize(6);
		display(); //основная функция
		glfwSwapBuffers(window);
		glfwPollEvents();
	} while (glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0 );
	
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	myCleanup();
	glfwTerminate();
	return 0;
}

void initGL() {
	//OpenGL - программный интерфейс для написания приложений, использующих двумерную и 
	//трёхмерную компьютерную графику. Независимый от языков программирования и ОС
	if( !glfwInit() )
	{
		fprintf( stderr, "Failed to initialize GLFW\n" );
		getchar();
		return;
	}
	//функция glfwWindowHint задает параметры для функции glfwCreateWindow
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); //задать версию клиентского API,
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); //с которым совместима данная программа
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); //для OpenGL версии 3 и выше
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE); //для какого профиля 
															//OpenGL создавать контекст

	window = glfwCreateWindow( window_width, window_height,
		"Template window", NULL, NULL);
	if( window == NULL ) {
		fprintf( stderr, "Failed to open GLFW window. \n" );
		getchar();
		glfwTerminate();
		return;
	}
	
	glfwMakeContextCurrent(window);
	//инициализация GLEW - библиотеки для упрощения загрузки расширений OpenGL
	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return;
	}
	return;
}
