import matplotlib.pyplot as plt
import random
import time
from fast_polynomial.polynomial import convolution, karat_driver, fft_driver


def benchmark(sizes: list[int], trials: int = 5) -> dict[str, list[float]]:
    timings: dict[str, list[float]] = {"convolution": [],
                                       "karatsuba": [],
                                       "fft": []}
    for n in sizes:
        for name, func in [("convolution", convolution),
                           ("karatsuba", karat_driver), ("fft", fft_driver)]:
            times = []
            for _ in range(trials):
                a = [random.random() for _ in range(n)]
                b = [random.random() for _ in range(n)]
                t1 = time.perf_counter()
                func(a, b)
                t2 = time.perf_counter()
                times.append(t2 - t1)
            timings[name].append(min(times))
    return timings


def plot(sizes: list[int], timings: dict[str, list[float]]) -> None:
    for name, times in timings.items():
        plt.plot(sizes, times, marker="o", label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Polynomial degree (n)")
    plt.ylabel("time (seconds)")
    plt.legend()
    plt.grid(linewidth=0.3)
    plt.show()


if __name__ == '__main__':
    # Performance Comparison Table
    print(f"{'Size (N)':<10} | {'Convolution (s)':<16} | {'C++ FFT (s)':<16} "
          + f"| {'Karatsuba (s)':<16} | {'Speedup':<10}")
    print("-" * 80)
    sizes: list[int] = [2**(i+3) for i in range(12)]
    timings = benchmark(sizes)

    for i, n in enumerate(sizes):
        t_convolution = timings["convolution"][i]
        t_karat = timings["karatsuba"][i]
        t_fft = timings["fft"][i]
        speedup = t_karat / t_fft if t_fft > 0 else 0
        print(f"{n:<10} | {t_convolution:<16.4f} | {t_fft:<16.4f} "
              + f"| {t_karat:<16.4f} | {speedup:<.1f}x")

    plot(sizes, timings)
