#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   
#include <vector>
#include <complex>
#include <cmath>

namespace py = pybind11;
const double PI = std::acos(-1.0);

// Recursive, out-of-place Cooley-Tukey FFT (requires a.size() to be a power of 2)
std::vector<std::complex<double>> cooley_tukey_fft(const std::vector<std::complex<double>>& a, bool invert) {
    size_t n = a.size();
    if (n <= 1) {
        return a;
    }

    // Split into even and odd indexed elements
    std::vector<std::complex<double>> even(n / 2);
    std::vector<std::complex<double>> odd(n / 2);
    for (size_t i = 0; i < n / 2; ++i) {
        even[i] = a[2 * i];
        odd[i]  = a[2 * i + 1];
    }

    // Divide
    std::vector<std::complex<double>> dft_even = cooley_tukey_fft(even, invert);
    std::vector<std::complex<double>> dft_odd  = cooley_tukey_fft(odd, invert);

    // Conquer
    std::vector<std::complex<double>> res(n);
    double angle = (invert ? 2.0 : -2.0) * PI / n;
    std::complex<double> twiddle(std::cos(angle), std::sin(angle));
    std::complex<double> w(1.0, 0.0);

    for (size_t k = 0; k < n / 2; ++k) {
        res[k] = dft_even[k] + w * dft_odd[k];
        res[k + n / 2] = dft_even[k] - w * dft_odd[k];
        w *= twiddle;
    }

    return res;
}

// Wrapper for Polynomial Multiplication via FFT Convolution
std::vector<double> multiply_polynomials(const std::vector<double>& poly1, const std::vector<double>& poly2) {
    if (poly1.empty() || poly2.empty()) return {};

    size_t target_size = poly1.size() + poly2.size() - 1;
    // Find the next power of 2 to satisfy Cooley-Tukey requirements
    size_t n = 1;
    while (n < target_size) {
        n *= 2;
    }

    // Starting arrays in time domain
    std::vector<std::complex<double>> ta(poly1.begin(), poly1.end());
    std::vector<std::complex<double>> tb(poly2.begin(), poly2.end());
    ta.resize(n, 0.0);
    tb.resize(n, 0.0);

    // FFT (frequency domain)
    std::vector<std::complex<double>> fa = cooley_tukey_fft(ta, false);
    std::vector<std::complex<double>> fb = cooley_tukey_fft(tb, false);
    std::vector<std::complex<double>> fproduct(n);

    // Pointwise multiplication in frequency domain
    for (size_t i = 0; i < n; ++i) {
        fproduct[i] = fa[i] * fb[i];
    }

    // Inverse FFT (time domain)
    std::vector<std::complex<double>> tproduct = cooley_tukey_fft(fproduct, true);

    // Normalize by 1/N for the inverse transform and extract real coefficients
    std::vector<double> result(target_size);
    for (size_t i = 0; i < target_size; ++i) {
        result[i] = tproduct[i].real() / static_cast<double>(n);
    }

    return result;
}

PYBIND11_MODULE(FFT, m) {
    m.doc() = "Simplified Cooley-Tukey FFT Polynomial Multiplier";
    m.def("multiply_polynomials", &multiply_polynomials, py::arg("poly1"), py::arg("poly2"), "Multiplies arrays via FFT Convolution");
}
