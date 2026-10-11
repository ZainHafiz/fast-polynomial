from typing import Union
from collections.abc import Iterable
from . import FFT


Number = Union[int, float]


"""Immutable univariate polynomial with three multiplication algorithms."""


class Polynomial:
    """
    Polynomial with coefficients a_i such that coeffs[i] is
    the coefficient of x**i
    """
    def __init__(self, coeffs: Iterable[Number]) -> None:
        """Initialise."""
        coeffs_tuple = tuple(coeffs)
        if not coeffs_tuple:
            coeffs_tuple = (0,)
        self._coeffs = coeffs_tuple
        self.__trim()

    def __trim(self) -> None:
        """
        Trim the coefficients of any leading zeros.

        This particular method allows for more efficient trimming since we
        don't need to repeatedly slice the tuple, it is only sliced once.
        """
        i = len(self._coeffs) - 1
        while i > 0 and self._coeffs[i] == 0:
            i -= 1
        self._coeffs = self._coeffs[: i + 1]

    @property
    def coeffs(self) -> tuple[Number]:
        """Return tuple of coefficients."""
        return self._coeffs

    @property
    def degree(self) -> int:
        """Return degree of polynomial."""
        return len(self.coeffs) - 1

    @property
    def leading(self) -> Number:
        return self.coeffs[-1]

    @property
    def is_zero(self) -> bool:
        return self.coeffs == (0,)

    def __str__(self) -> str:
        """Return a readable string representation, highest degree first."""
        if self.is_zero:
            return "0"

        parts: list[str] = []
        for d in range(self.degree, -1, -1):  # Iterate from highest degree
            c = self.coeffs[d]
            if not c:
                continue

            # ---- sign ----
            if not parts:
                sign = "-" if c < 0 else ""
            else:
                sign = " - " if c < 0 else " + "

            # ---- coefficient string ----
            abs_c = abs(c)
            if abs_c == 1 and d != 0:
                coeff_str = ""
            else:
                if abs_c % 1 == 0:
                    coeff_str = str(int(abs(c)))
                else:
                    coeff_str = str(abs(c))

            # ---- variable / power ----
            if d == 1:
                var_str = "x"
            elif d == 0:
                var_str = ""
            else:
                var_str = f"x^{d}"
            parts.append(f"{sign}{coeff_str}{var_str}")

        return "".join(parts)

    def __repr__(self) -> str:
        return f"Polynomial({self.coeffs!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Polynomial):
            return self.coeffs == other.coeffs
        else:
            return False

    def __neg__(self) -> "Polynomial":
        return -1*self

    def __call__(self, x: Number) -> Number:
        """Implement Horner method for evaluating polynomials efficiently."""

        result: Number = 0
        for c in reversed(self.coeffs):
            result = result * x + c
        return result

    def __add__(self, other: 'Polynomial' | Number) -> "Polynomial":
        """Add two polynomials or a polynomial and a scalar."""

        if isinstance(other, Polynomial):
            coeff_sum = [a + b for a, b in zip(self.coeffs, other.coeffs)]
            index = min(self.degree + 1, other.degree + 1)
            coeff_sum += self.coeffs[index:] + other.coeffs[index:]
            return Polynomial(coeff_sum)
        elif isinstance(other, Number):
            return Polynomial((self.coeffs[0] + other,) + self.coeffs[1:])
        else:
            return NotImplemented

    def __radd__(self, other: Number) -> "Polynomial":
        return self + other

    def __sub__(self, other: 'Polynomial' | Number) -> "Polynomial":
        """Subtract two polynomials or a scalar from polynomial."""

        if isinstance(other, Polynomial):
            dif = [a - b for a, b in zip(self.coeffs, other.coeffs)]
            index = min(self.degree + 1, other.degree + 1)
            dif += self.coeffs[index:] + (-other).coeffs[index:]
            return Polynomial(dif)
        elif isinstance(other, Number):
            return Polynomial((self.coeffs[0] - other,) + self.coeffs[1:])
        else:
            return NotImplemented

    def __rsub__(self, other: Number) -> "Polynomial":
        return (-self) + other

    def __mul__(self, other: 'Polynomial' | Number) -> "Polynomial":
        """
        Multiply two polynomials or a Polynomials and a scalar

        Does this via Cooley-Tukey FFT algorithm for large degrees, the
        Karatsuba method for smaller degrees, and a convolution for smaller
        degrees.
        """

        if isinstance(other, Polynomial):
            if self.is_zero or other.is_zero:
                return Polynomial(())
            if self.degree == 0:
                return other * self.leading
            if other.degree == 0:
                return self * other.leading
            KARATSUBA_THRESHOLD = 500
            FFT_THRESHOLD = 1000
            if (len(self.coeffs) < KARATSUBA_THRESHOLD or len(other.coeffs) <
               KARATSUBA_THRESHOLD):
                return Polynomial(convolution(list(self.coeffs),
                                              list(other.coeffs)))

            if (len(self.coeffs) <= FFT_THRESHOLD or len(other.coeffs) <=
               FFT_THRESHOLD):
                return Polynomial(karat_driver(list(self.coeffs),
                                               list(other.coeffs)))

            return Polynomial(fft_driver(list(self.coeffs),
                                         list(other.coeffs)))
        elif isinstance(other, Number):
            return Polynomial(other * c for c in self.coeffs)
        else:
            return NotImplemented

    def __rmul__(self, other: Number) -> "Polynomial":
        return self * other

    def multiply(self, other: 'Polynomial',
                 method: str = "auto") -> "Polynomial":
        """
        Multiply two polynomials together allowing for choice of algorithms

        method must be string among {"auto", "convolution", "karatsuba", "fft"}
        , where auto calls __mul__ multiplication which changes algorithm
        based on degree thresholds.
        """
        if isinstance(other, Polynomial):
            match method:
                case "auto":
                    return self * other
                case "convolution":
                    return Polynomial(convolution(list(self.coeffs),
                                                  list(other.coeffs)))
                case "karatsuba":
                    return Polynomial(karat_driver(list(self.coeffs),
                                                   list(other.coeffs)))
                case "fft":
                    return Polynomial(fft_driver(list(self.coeffs),
                                                 list(other.coeffs)))
                case _:
                    if isinstance(method, str):
                        raise ValueError('method must be among keywords '
                                         + '{"auto", "convolution", '
                                         + '"karatsuba", "fft"}')
                    raise TypeError('method must be of type string, and among '
                                    + 'the keywords {"auto", "convolution", '
                                    + '"karatsuba", "fft"}')
        elif isinstance(other, Number):
            return self * other
        else:
            raise TypeError("other argument must be of type Polynomial, "
                            + "int or float")

    def __pow__(self, exponent: int) -> "Polynomial":
        """Return the polynomial raised to a non-negative integer power.

        Uses exponentiation by squaring, requiring O(log k) multiplications
        where k is the exponent.
        """
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be integer, got a"
                            + f"{type(exponent)} instead.")

        if exponent < 0:
            raise ValueError("Exponent must be a non-negative integer.")

        if exponent == 0:
            return Polynomial((1,))

        result = Polynomial((1,))
        base = self
        n = exponent

        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2

        return result

    def __divmod__(self, other: 'Polynomial' | Number) -> tuple["Polynomial",
                                                                "Polynomial"]:
        """Implement polynomial division, returning quotient and remainder."""
        if isinstance(other, Number):
            if not other:
                raise ZeroDivisionError(
                    "Cannot divide polynomial by zero scalar.")
            return (Polynomial(coeff / other for coeff in self.coeffs),
                    Polynomial((0,)))
        if not isinstance(other, Polynomial):
            return NotImplemented
        if other.degree == 0:
            if other.is_zero:
                raise ZeroDivisionError(
                    "Cannot divide polynomial by the zero polynomial.")
            return (Polynomial(coeff / other.leading for coeff in self.coeffs),
                    Polynomial((0,)))

        if self.is_zero:
            return Polynomial((0,)), Polynomial((0,))
        numerator = Polynomial(self.coeffs)
        denominator = other
        quotient_coeffs = []
        x = Polynomial((0, 1))
        while numerator.degree >= denominator.degree:
            difference = (numerator.degree - denominator.degree)
            quotient_coeffs.append(numerator.coeffs[-1]/denominator.leading)
            numerator -= quotient_coeffs[-1] * x**(difference) * denominator
        return (Polynomial(reversed(quotient_coeffs)), numerator)

    def __floordiv__(self, other: 'Polynomial' | Number) -> "Polynomial":
        """Return quotient of polynomial division, self // other."""
        if isinstance(other, Number):
            if other == 0:
                raise ZeroDivisionError(
                    "Cannot divide polynomial by zero scalar.")
            return Polynomial(coeff // other for coeff in self.coeffs)
        if not isinstance(other, Polynomial):
            return NotImplemented
        if other.degree == 0:
            if other.is_zero:
                raise ZeroDivisionError(
                    "Cannot divide polynomial by the zero polynomial.")
            return Polynomial(coeff // other.leading for coeff in self.coeffs)
        q, _ = divmod(self, other)
        return q

    def __mod__(self, other: 'Polynomial' | Number) -> "Polynomial":
        """Return remainder of polynomial division, self % other."""
        if isinstance(other, Number):
            if other == 0:
                raise ZeroDivisionError(
                    "Cannot divide polynomial by scalar zero.")
            return Polynomial((0,))
        _, r = divmod(self, other)
        return r

    def derivative(self) -> "Polynomial":
        coeffs = tuple((d+1)*c for d, c in enumerate(self.coeffs[1:]))
        return Polynomial(coeffs)

    def integral(self, y_0: Number = 0) -> "Polynomial":
        """Return an antiderivative of the polynomial.

        The constant term is chosen so that the resulting polynomial satisfies
        F(0) = y_0 (default 0).
        """
        return Polynomial((y_0,) + tuple(c/(d+1) for d, c in
                                         enumerate(self.coeffs)))


def convolution(p: list[Number], q: list[Number]) -> list[Number]:
    """
    Implements convolution multiplication of polynomials.
    """
    product: list[Number] = [0]*(len(p)+len(q)-1)
    for i, p_i in enumerate(p):
        for j, q_j in enumerate(q):
            product[i+j] += p_i * q_j
    return product


def karat_driver(p: list[Number], q: list[Number]) -> list[Number]:
    """
    Provides padding logic for Karatsuba method.
    """
    def karat_mult_poly(p: list[Number], q: list[Number]) -> list[Number]:
        """
        Implements Karatsuba algorithm for polynomial multiplication

        Assume both p and q have equal power of 2 lengths.
        """
        n = len(p)  # p and q degree = n-1
        if n == 1:
            return [p[0]*q[0]]

        m = n//2
        # Divide
        p0, p1, q0, q1 = p[:m], p[m:], q[:m], q[m:]  # each len = m
        p_sum = [a + b for a, b in zip(p0, p1)]  # len = m
        q_sum = [a + b for a, b in zip(q0, q1)]  # len = m

        # Conquer
        r1 = karat_mult_poly(p0, q0)  # len = 2*m - 1, degree = 2*m
        r2 = karat_mult_poly(p1, q1)  # len = 2*m - 1
        r3 = karat_mult_poly(p_sum, q_sum)  # len = 2*m - 1
        r4 = [r3[i] - r1[i] - r2[i] for i in range(2*m - 1)]

        r: list[Number] = [0]*(2*n-1)  # len = 4*m - 1, degree = 2*n - 2
        for i in range(2*m-1):
            r[i] += r1[i]
            r[i+m] += r4[i]
            r[i+2*m] += r2[i]

        return r

    max_len = max(len(p), len(q))
    if max_len & (max_len - 1) == 0:
        n = max_len
    else:
        n = 1 << max_len.bit_length()

    padded_p = p + [0]*(n - len(p))
    padded_q = q + [0]*(n - len(q))
    return karat_mult_poly(padded_p, padded_q)


def fft_driver(p: list[Number], q: list[Number]) -> list[Number]:
    result = FFT.multiply_polynomials(poly1=p, poly2=q)
    result = [round(x, 6) for x in result]
    return result
