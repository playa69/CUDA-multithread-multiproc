
# README

## Описание

Этот проект демонстрирует различные методы умножения элементов массива на константу, используя:

- Многопоточность
- Многопроцессность
- CUDA
- Numpy

## Установка

1. Установите необходимые библиотеки:
    ```bash
    pip install numpy tqdm psutil numba
    ```

## Запуск

Для запуска кода используйте команду:
```bash
python script.py
```


## Методы
### Многопоточность
Код использует threading для параллельного выполнения задач. Это полезно для задач, которые могут быть разделены на независимые части. Однако, из-за GIL (Global Interpreter Lock) в CPython, многопоточность не всегда эффективна для вычислительных задач.

### Многопроцессность
Использование multiprocessing позволяет обходить ограничения GIL, создавая отдельные процессы для выполнения задач. Это увеличивает использование CPU для вычислительно интенсивных задач.

### CUDA
С помощью библиотеки numba и CUDA можно выполнять вычисления на GPU, что значительно ускоряет обработку больших массивов данных. Это полезно для задач, которые могут быть параллелизированы на уровне GPU.

### Numpy
Использование библиотеки numpy позволяет эффективно выполнять операции над массивами благодаря оптимизированным подкапотным реализациям. Это делает Numpy очень мощным инструментом для научных вычислений и обработки данных.

## Output:

==== Testing naive iterative method ====

Average CPU utilization: 100%
Iterative time: 4.532145023345947s

==== Testing using numpy ====

Average CPU utilization: 100%
Numpy time: 0.05124187469482422s

==== Testing with threads ====

Average CPU utilization: 100%
Multithreaded time: 2.3054239749908447s

==== Testing with processes ====

Average CPU utilization: 100%
Multiprocessed time: 1.4523680210113525s

==== Testing with CUDA ====

GPU devices: <CUDA Device 0 'NVIDIA GeForce GTX 1060'>
Average CPU utilization: 100%
0.003217935562133789s

==== Testing matrix multiplication with CUDA ====

CUDA multime: 0.015402793884277344
[[499500000000 499500000000 ...]
 [499500000000 499500000000 ...]
 ...]

==== Testing matrix multiplication with NUMPY ====

NUMPY multime: 0.03211402893066406
