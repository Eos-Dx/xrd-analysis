import multiprocessing


def stress():
    # Busy loop that does nothing but consume CPU cycles
    while True:
        pass


if __name__ == '__main__':
    # Determine the number of CPU cores available
    num_cores = multiprocessing.cpu_count()
    print(f"Starting CPU stress test on {num_cores} cores.")

    processes = []
    for _ in range(num_cores):
        p = multiprocessing.Process(target=stress)
        p.start()
        processes.append(p)

    # Optionally join the processes to keep the script running
    for p in processes:
        p.join()
