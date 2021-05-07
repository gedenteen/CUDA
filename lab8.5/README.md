Для запуска на Ubuntu нужно проделать следующие шаги:
1. Проверить, что python3 установлен:
```
$ python3 --version
``` 
2. Установить Anaconda (дистрибутив Python) - скачать sh-файл с сайта и запустить его:
```
$ bash *.sh
``` 
В конце установки написать "yes", чтобы Conda запускалась автоматически при запуске терминала.
3. https://developer.nvidia.com/how-to-cuda-python
4. Установить пакеты Python https://pyprog.pro/installing_numpy.html


* just_python.py - отображение Мандельброта c помощью обычного Питона
* python_numba.py - отображение Мандельброта c помощью Питона и numba
* python_numba_cuda.py - отображение Мандельброта c помощью Питона, numba и cuda
* cuda_c.cu - отображение Мандельброта с помощью cuda C
* plot_for_cuda_c.py - программа на Питоне для постороения картинки на основе данных cuda_c.cu
* other_progs/e4.py - Шлирен метод
