#include "task1.h"
#include "task2.h"
#include "task3.h"

int main()
{
    cudaDeviceProp prop;
    int whichDevice;

    // проверяем поддерживает ли устройство overlapping computation with memory copy
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        printf("device will not handle\n");
        return 0;
    }
    
    //srand(time(0));
    task1();
    task2();
    task3();
    
    return 0;
}

