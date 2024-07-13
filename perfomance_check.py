import numpy as np
import time
import tqdm
import psutil
import math
import threading
import multiprocessing
from numba import cuda

# Класс для мониторинга ресурсов
class ResourceMonitor:
    def __init__(self):
        self._monitor_resources = False
        self._avg_cpu = None

    def start_monitoring(self):
        self._monitor_resources = True
        self._avg_cpu = []

    def stop_monitoring(self):
        self._monitor_resources = False

    def get_average_cpu(self):
        return int(sum(self._avg_cpu) / len(self._avg_cpu)) if self._avg_cpu else None

    def monitor(self):
        with tqdm.tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm.tqdm(total=100, desc='ram%', position=0) as rambar:
            while self._monitor_resources:
                rambar.n = psutil.virtual_memory().percent
                self._avg_cpu.append(psutil.cpu_percent())
                cpubar.n = self._avg_cpu[-1]
                rambar.refresh()
                cpubar.refresh()
                time.sleep(0.001)

# Декоратор для измерения времени выполнения функций
class TimeCounter:
    def __init__(self, monitor):
        self.monitor = monitor

    def __call__(self, f):
        def wrapped(*args, **kwargs):
            self.monitor.start_monitoring()
            monitor_thread = threading.Thread(target=self.monitor.monitor)
            monitor_thread.start()
            start_time = time.time()
            result = f(*args, **kwargs)
            end_time = time.time()
            self.monitor.stop_monitoring()
            monitor_thread.join()

            avg_cpu = self.monitor.get_average_cpu()
            print(f"Average CPU utilization: {avg_cpu}%")
            return result, end_time - start_time
        return wrapped

# Методы умножения
class Multipliers:
    @staticmethod
    @cuda.jit
    def cuda_multiply_by_2(array):
        pos = cuda.grid(1)
        if pos < array.size:
            array[pos] *= 2

    @staticmethod
    def multiplier_mock(i: int, batch: int):
        for j in range(i, i + batch):
            x = 1
            x *= 2

    @staticmethod
    @TimeCounter(ResourceMonitor())
    def threaded_multiply_by_2(array: list, indexes: list, batch: int, use_processes=False) -> list:
        threads = []
        for i in indexes:
            threads.append(
                multiprocessing.Process(target=Multipliers.multiplier_mock, args=[i, batch])
                if use_processes
                else threading.Thread(target=Multipliers.multiplier_mock, args=[i, batch])
            )
            threads[-1].start()
        for tr in threads:
            tr.join()
        return array

    @staticmethod
    @TimeCounter(ResourceMonitor())
    def numpy_multiply_by_2(array: np.array) -> np.array:
        array *= 2
        return array

    @staticmethod
    @TimeCounter(ResourceMonitor())
    def iterative_multiply_by_2(array: list) -> list:
        return [i * 2 for i in array]

    @staticmethod
    @cuda.jit
    def matmul(A, B, C):
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

# Вспомогательная функция для создания массива
def create_array():
    return [1] * 100000000

if __name__ == "__main__":
    monitor = ResourceMonitor()
    time_counter = TimeCounter(monitor)

    print("\n==== Testing naive iterative method ====\n")
    array = create_array()
    _, iterative_time = Multipliers.iterative_multiply_by_2(array)
    print(f"Iterative time: {iterative_time}s")

    print("\n==== Testing using numpy ====\n")
    array = np.array(create_array())
    _, numpy_time = Multipliers.numpy_multiply_by_2(array)
    print(f"Numpy time: {numpy_time}s")

    print("\n==== Testing with threads ====\n")
    array = create_array()
    _, threads_time = Multipliers.threaded_multiply_by_2(array, range(0, len(array), 10 ** 7), 10 ** 7)
    print(f"Multithreaded time: {threads_time}s")

    print("\n==== Testing with processes ====\n")
    array = create_array()
    _, processes_time = Multipliers.threaded_multiply_by_2(array, range(0, len(array), 10 ** 7), 10 ** 7, use_processes=True)
    print(f"Multiprocessed time: {processes_time}s")

    print("\n==== Testing with CUDA ====\n")
    print(f"GPU devices: {cuda.gpus}")
    array = np.array(create_array())
    threadsperblock = 32
    blockspergrid = (array.size + (threadsperblock - 1)) // threadsperblock
    _, cuda_time = time_counter(Multipliers.cuda_multiply_by_2[blockspergrid, threadsperblock])(array)
    print(f"{cuda_time}")

    print("\n==== Testing matrix multiplication with CUDA ====\n")
    x_h = np.arange(1000000).reshape([1000, 1000])
    y_h = np.arange(1000000).reshape([1000, 1000])
    z_h = np.zeros([1000, 1000])

    x_d = cuda.to_device(x_h)
    y_d = cuda.to_device(y_h)
    z_d = cuda.to_device(z_h)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(z_h.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(z_h.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _, cuda_time2 = time_counter(Multipliers.matmul[blockspergrid, threadsperblock])(x_d, y_d, z_d)
    print(f"CUDA multime: {cuda_time2}")
    z_h = z_d.copy_to_host()
    print(z_h)

    print("\n==== Testing matrix multiplication with NUMPY ====\n")
    _, numpy_time2 = time_counter(np.matmul)(x_h, y_h)
    print(f"NUMPY multime: {numpy_time2}")
