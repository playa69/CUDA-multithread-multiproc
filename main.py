import numpy
import time
import tqdm
import psutil

import math
import threading
import multiprocessing
import time

from numba import cuda

_MONITOR_RESOURCES = False
_AVG_CPU = None

_SMALL_INFO_ABOUT_GIL = """
Note that in many cases (and virtually all cases where your "expensive operation" is a calculation implemented in Python), multiple threads will not actually run concurrently due to Python's Global Interpreter Lock (GIL).
The GIL is an interpreter-level lock. This lock prevents execution of multiple threads at once in the Python interpreter. Each thread that wants to run must wait for the GIL to be released by the other thread, which means your multi-threaded Python application is essentially single threaded, right? Yes. Not exactly. Sort of.
CPython uses what’s called “operating system” threads under the covers, which is to say each time a request to make a new thread is made, the interpreter actually calls into the operating system’s libraries and kernel to generate a new thread. This is the same as Java, for example. So in memory you really do have multiple threads and normally the operating system controls which thread is scheduled to run. On a multiple processor machine, this means you could have many threads spread across multiple processors, all happily chugging away doing work.
However, while CPython does use operating system threads (in theory allowing multiple threads to execute within the interpreter simultaneously), the interpreter also forces the GIL to be acquired by a thread before it can access the interpreter and stack and can modify Python objects in memory all willy-nilly. The latter point is why the GIL exists: The GIL prevents simultaneous access to Python objects by multiple threads. But this does not save you (as illustrated by the Bank example) from being a lock-sensitive creature; you don’t get a free ride. The GIL is there to protect the interpreters memory, not your sanity.
"""


def monitor():
    global _AVG_CPU
    with tqdm.tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm.tqdm(total=100, desc='ram%',
                                                                            position=0) as rambar:
        cpu_percent = []
        while _MONITOR_RESOURCES:
            rambar.n = psutil.virtual_memory().percent
            cpu_percent.append(psutil.cpu_percent())
            cpubar.n = cpu_percent[-1]
            rambar.refresh()
            cpubar.refresh()
            time.sleep(0.001)
        _AVG_CPU = int(sum(cpu_percent) / len(cpu_percent))


def time_counter(f):
    def decorated(*args, **kwargs):
        global _MONITOR_RESOURCES
        _MONITOR_RESOURCES = True
        mons = threading.Thread(target=monitor)
        mons.start()
        t1 = time.time()
        result = f(*args, **kwargs)
        t2 = time.time()
        _MONITOR_RESOURCES = False
        mons.join()

        print(f"Average cpu utilization: {_AVG_CPU}%")

        return result, t2 - t1

    return decorated


@cuda.jit
def cuda_multiply_by_2(array):
    pos = cuda.grid(1)
    if pos < array.size:
        array[pos] *= 2


def _multiplier_mock(i: int, batch: int):
    '''
    I have mocked multiplier for multiprocessing solution to disable time of copying array to own process virtual memory
    Just testing compute perf
    '''
    for j in range(i, i + batch):
        x = 1
        x *= 2


@time_counter
def threaded_multiply_by_2(array: list, indexes: list, batch: int, use_processes=False) -> list:
    threads = []
    for i in indexes:
        threads.append(
            multiprocessing.Process(target=_multiplier_mock, args=[i, batch])
            if use_processes
            else threading.Thread(target=_multiplier_mock, args=[i, batch])
        )
        threads[-1].start()
    for tr in threads: tr.join()
    return array


@time_counter
def numpy_multiply_by_2(array: numpy.array) -> numpy.array:
    array *= 2
    return array


@time_counter
def iterative_multiply_by_2(array: list) -> list:
    return [i * 2 for i in array]


def create_array():
    return [1] * 100000000


@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B."""
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


if __name__ == "__main__":


    print("\n==== Testing naive iterative method ====\n")
    array = create_array()
    _, iterative_time = iterative_multiply_by_2(array)
    print(f"Iterative time: {iterative_time}s")

    print("\n==== Testing using numpy ====\n")
    array = numpy.array(create_array())
    _, numpy_time = numpy_multiply_by_2(array)
    print(f"Numpy time: {numpy_time}s")

    print("\n==== Testing with threads ====\n")
    array = create_array()
    _, threads_time = threaded_multiply_by_2(array, range(0, len(array), 10 ** 7), 10 ** 7)
    print(f"Multithreaded time: {threads_time}s")
    print(f"Strange result happened cause of: \n{_SMALL_INFO_ABOUT_GIL}")

    print("\n==== Testing with processes ====\n")
    array = create_array()
    _, threads_time = threaded_multiply_by_2(array, range(0, len(array), 10 ** 7), 10 ** 7, use_processes=True)
    print(f"Multiprocessed time: {threads_time}s")

    print("\n==== Testing with CUDA ====\n")
    print(f"GPU devices: {cuda.gpus}")
    array = numpy.array(create_array())
    threadsperblock = 32
    blockspergrid = (array.size + (threadsperblock - 1)) // threadsperblock
    _,cuda_time = time_counter(cuda_multiply_by_2[blockspergrid, threadsperblock])(array)
    print(f"{cuda_time}")



    print("\n==== Testing matrix multiplication with CUDA ====\n")


    #x_h = numpy.arange(16).reshape([4, 4])
    #y_h = numpy.ones([4, 4])
    #z_h = numpy.zeros([4, 4])

    x_h = numpy.arange(1000000).reshape([1000, 1000])
    y_h = numpy.arange(1000000).reshape([1000, 1000])
    z_h = numpy.zeros([1000, 1000])

    print("\n A: ")
    print(x_h)
    print("\n B: ")
    print(y_h)
    print("\n C: ")
    print(z_h)

    print("\n <<-------->> \n ")

    x_d = cuda.to_device(x_h)
    y_d = cuda.to_device(y_h)
    z_d = cuda.to_device(z_h)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(z_h.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(z_h.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    ## _ , cuda_time2 = time_counter(matmul[blockspergrid, threadsperblock])(x_d, y_d, z_d)

    _, cuda_time2 = time_counter(matmul[blockspergrid, threadsperblock])(x_d, y_d, z_d)



    print(f"CUDA multime: {cuda_time2}")
    z_h = z_d.copy_to_host()
    print(z_h)
    print("\n-------------------------\n")
    _, nump_time = time_counter( numpy.matmul )(x_h, y_h)


    print("\n==== Testing matrix multiplication with NUMPY ====\n")
    ##print(f"NUMPY multime: {x_h @ y_h}")
    print(f"NUMPY multime: {nump_time}")




### x_h = np.arange(16).reshape([4, 4])
### y_h = np.ones([4, 4])
### z_h = np.zeros([4, 4])
###
### x_d = cuda.to_device(x_h)
### y_d = cuda.to_device(y_h)
### z_d = cuda.to_device(z_h)
###
### threadsperblock = (16, 16)
### blockspergrid_x = math.ceil(z_h.shape[0] / threadsperblock[0])
### blockspergrid_y = math.ceil(z_h.shape[1] / threadsperblock[1])
### blockspergrid = (blockspergrid_x, blockspergrid_y)
###
### matmul[blockspergrid, threadsperblock](x_d, y_d, z_d)
### z_h = z_d.copy_to_host()
### print(z_h)
### print(x_h @ y_h)




   #print("\n==== Testing naive iterative method ====\n")
   #array = create_array()
   #_, iterative_time = iterative_multiply_by_2(array)
   #print(f"Iterative time: {iterative_time}s")

   #print("\n==== Testing using numpy ====\n")
   #array = numpy.array(create_array())
   #_, numpy_time = numpy_multiply_by_2(array)
   #print(f"Numpy time: {numpy_time}s")

   #print("\n==== Testing with threads ====\n")
   #array = create_array()
   #_, threads_time = threaded_multiply_by_2(array, range(0, len(array), 10 ** 7), 10 ** 7)
   #print(f"Multithreaded time: {threads_time}s")
   #print(f"Strange result happened cause of: \n{_SMALL_INFO_ABOUT_GIL}")

   #print("\n==== Testing with processes ====\n")
   #array = create_array()
   #_, threads_time = threaded_multiply_by_2(array, range(0, len(array), 10 ** 7), 10 ** 7, use_processes=True)
   #print(f"Multiprocessed time: {threads_time}s")