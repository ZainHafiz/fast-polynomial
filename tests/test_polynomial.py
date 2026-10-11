import pytest
from fast_polynomial import Polynomial


# -----------------------------
# Helpers
# -----------------------------

def poly(coeffs):
    """Small helper to make tests shorter."""
    return Polynomial(coeffs)


# -----------------------------
# Construction & basic properties
# -----------------------------

@pytest.mark.parametrize(
    "coeffs, expected_coeffs",
    [
        ((0,), (0,)),
        ((0, 0, 0), (0,)),
        ((1, 2, 0, 0), (1, 2)),
        ((3,), (3,)),
        ((3, 0, 0), (3,)),
    ],
)
def test_construction_trimming(coeffs, expected_coeffs):
    p = poly(coeffs)
    assert p.coeffs == expected_coeffs


def test_degree_basic():
    assert poly((0,)).degree == 0
    assert poly((1,)).degree == 0
    assert poly((1, 2)).degree == 1
    assert poly((1, 2, 3)).degree == 2


@pytest.mark.parametrize(
    "a, b, equal",
    [
        ((1, 2, 3), (1, 2, 3), True),
        ((1, 2), (1, 2, 0, 0), True),   # trimming should make these equal
        ((0,), (0, 0, 0), True),
        ((2,), (3,), False),
    ],
)
def test_equality_and_trimming(a, b, equal):
    p = poly(a)
    q = poly(b)
    assert (p == q) is equal


# -----------------------------
# String representation
# -----------------------------

@pytest.mark.parametrize(
    "coeffs, expected",
    [
        ((0,), "0"),
        ((3,), "3"),
        ((-3,), "-3"),
        ((0, 1), "x"),
        ((1, 1), "x + 1"),
        ((0, 1, 2, 3), "3x^3 + 2x^2 + x"),
        ((1, -2, 0, 3), "3x^3 - 2x + 1"),
    ],
)
def test_str(coeffs, expected):
    p = poly(coeffs)
    assert str(p) == expected


def test_repr_roundtrip():
    p = poly((1, -2, 3))
    r = repr(p)
    # Simple sanity: repr should contain class name and coeffs
    assert "Polynomial" in r
    assert "(1, -2, 3)" in r


# -----------------------------
# Evaluation (__call__)
# -----------------------------

@pytest.mark.parametrize(
    "coeffs, x, expected",
    [
        # 3x^2 - 5x + 2
        ((2, -5, 3), 0, 2),
        ((2, -5, 3), 1, 0),
        ((2, -5, 3), 2, 4),
        # constant
        ((5,), 10, 5),
        # linear: -2x + 1
        ((1, -2), -1, 3),
    ],
)
def test_call_evaluation(coeffs, x, expected):
    p = poly(coeffs)
    assert p(x) == expected


# -----------------------------
# Addition & subtraction
# -----------------------------

@pytest.mark.parametrize(
    "a, b, expected",
    [
        ((1, 2), (3, -2, 4), (4, 0, 4)),
        ((0,), (0,), (0,)),
        ((1, 2, 3), (-1, -2, -3), (0,)),
    ],
)
def test_addition_poly_poly(a, b, expected):
    p = poly(a)
    q = poly(b)
    r = p + q
    assert r.coeffs == expected


@pytest.mark.parametrize(
    "p_coeffs, scalar, expected",
    [
        ((3, -2, 4), 5, (8, -2, 4)),   # p + c
        ((3, -2, 4), -3, (0, -2, 4)),
    ],
)
def test_scalar_addition_right(p_coeffs, scalar, expected):
    p = poly(p_coeffs)
    r = p + scalar
    assert r.coeffs == expected


@pytest.mark.parametrize(
    "scalar, p_coeffs, expected",
    [
        (5, (3, -2, 4), (8, -2, 4)),   # c + p
        (-3, (3, -2, 4), (0, -2, 4)),
    ],
)
def test_scalar_addition_left(scalar, p_coeffs, expected):
    p = poly(p_coeffs)
    r = scalar + p
    assert r.coeffs == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ((1, 2), (3, -2, 4), (-2, 4, -4)),
        ((3, -2, 4), (3, -2, 4), (0,)),
    ],
)
def test_subtraction_poly_poly(a, b, expected):
    p = poly(a)
    q = poly(b)
    r = p - q
    assert r.coeffs == expected


@pytest.mark.parametrize(
    "p_coeffs, scalar, expected",
    [
        ((3, -2, 4), 5, (-2, -2, 4)),   # p - c
        ((3, -2, 4), -3, (6, -2, 4)),
    ],
)
def test_scalar_subtraction_right(p_coeffs, scalar, expected):
    p = poly(p_coeffs)
    r = p - scalar
    assert r.coeffs == expected


@pytest.mark.parametrize(
    "scalar, p_coeffs, expected",
    [
        (5, (3, -2, 4), (2, 2, -4)),   # 5 - (4x^2 - 2x + 3) = -4x^2 + 2x + 2
    ],
)
def test_scalar_subtraction_left(scalar, p_coeffs, expected):
    p = poly(p_coeffs)
    r = scalar - p
    assert r.coeffs == expected


# -----------------------------
# Multiplication
# -----------------------------

def test_multiplication_poly_poly():
    p = poly((1, 1))      # x + 1
    q = poly((1, -1))     # -x + 1
    r = p * q             # (x + 1)(-x + 1) = -x^2 + 1
    assert r.coeffs == (1, 0, -1)


@pytest.mark.parametrize(
    "p_coeffs, scalar, expected",
    [
        ((2, -1, 3), 4, (8, -4, 12)),
        ((2, -1, 3), -1, (-2, 1, -3)),
        ((0,), 5, (0,)),
    ],
)
def test_scalar_multiplication_right(p_coeffs, scalar, expected):
    p = poly(p_coeffs)
    r = p * scalar
    assert r.coeffs == expected


@pytest.mark.parametrize(
    "scalar, p_coeffs, expected",
    [
        (4, (2, -1, 3), (8, -4, 12)),
        (-1, (2, -1, 3), (-2, 1, -3)),
    ],
)
def test_scalar_multiplication_left(scalar, p_coeffs, expected):
    p = poly(p_coeffs)
    r = scalar * p
    assert r.coeffs == expected


def test_multiplication_by_zero_poly():
    p = poly((1, 2, 3))
    z = poly((0,))
    assert (p * z).coeffs == (0,)
    assert (z * p).coeffs == (0,)


# -----------------------------
# Powers
# -----------------------------

def test_power_0():
    p = poly((2, 3))
    assert (p ** 0).coeffs == (1,)


def test_power_1():
    p = poly((2, 3))
    assert (p ** 1) == p


def test_power_3():
    p = poly((1, 1))  # x + 1
    # (x + 1)^3 = x^3 + 3x^2 + 3x + 1
    assert (p ** 3).coeffs == (1, 3, 3, 1)


def test_power_negative_raises():
    p = poly((1, 1))
    with pytest.raises(ValueError):
        p ** -1


def test_power_non_integer_raises():
    p = poly((1, 1))
    with pytest.raises(TypeError):
        p ** 2.5


# -----------------------------
# Derivative & Integral
# -----------------------------

def test_derivative_quadratic():
    p = poly((2, -5, 3))  # 3x^2 - 5x + 2
    dp = p.derivative()   # 6x - 5
    assert dp.coeffs == (-5, 6)


def test_derivative_constant():
    p = poly((5,))
    dp = p.derivative()
    assert dp.coeffs == (0,)


def test_integral_with_constant():
    p = poly((2, -5, 3))   # 3x^2 - 5x + 2
    F = p.integral(5)      # x^3 - (5/2)x^2 + 2x + 5
    # coefficients: [5, 2, -5/2, 1]
    assert F.coeffs == (5, 2, -2.5, 1)


def test_integral_default():
    p = poly((3,))         # 3
    F = p.integral()       # 3x
    assert F.coeffs == (0, 3)


# -----------------------------
# Division: divmod, //, %
# -----------------------------

def test_divmod_simple():
    # (x^2 - 1) / (x - 1) = x + 1, remainder 0
    p = poly((-1, 0, 1))   # x^2 - 1
    q = poly((-1, 1))      # x - 1
    quotient, remainder = divmod(p, q)
    assert quotient.coeffs == (1, 1)   # x + 1
    assert remainder.coeffs == (0,)    # 0


def test_divmod_nonzero_remainder():
    # (x^2 + 1) / (x - 1) = x + 1, remainder 2
    p = poly((1, 0, 1))    # x^2 + 1
    q = poly((-1, 1))      # x - 1
    quotient, remainder = divmod(p, q)
    assert quotient.coeffs == (1, 1)   # x + 1
    assert remainder.coeffs == (2,)    # 2


def test_floordiv_and_mod():
    p = poly((1, 0, 1))    # x^2 + 1
    q = poly((-1, 1))      # x - 1
    assert (p // q).coeffs == (1, 1)
    assert (p % q).coeffs == (2,)


def test_divide_by_constant_polynomial():
    p = poly((2, 4, 6))    # 2 + 4x + 6x^2
    c = poly((2,))         # 2
    q, r = divmod(p, c)
    assert q.coeffs == (1, 2, 3)
    assert r.coeffs == (0,)


def test_divmod_zero_dividend():
    p = poly((0,))
    q = poly((1, 2))
    quotient, remainder = divmod(p, q)
    assert quotient.coeffs == (0,)
    assert remainder.coeffs == (0,)


def test_division_by_zero_polynomial_raises():
    p = poly((1, 2))
    z = poly((0,))
    with pytest.raises(ZeroDivisionError):
        divmod(p, z)


def test_division_by_zero_scalar_raises():
    p = poly((1, 2, 3))
    with pytest.raises(ZeroDivisionError):
        divmod(p, 0)


def test_mod_by_scalar_zero_raises():
    p = poly((1, 2, 3))
    with pytest.raises(ZeroDivisionError):
        _ = p % 0


def test_divmod_by_scalar():
    p = poly((2, 4, 6))
    q, r = divmod(p, 2)
    assert q.coeffs == (1, 2, 3)
    assert r.coeffs == (0,)


# -----------------------------
# Multiplication algorithms (convolution, karatsuba, fft)
# -----------------------------

# Known small products, checked by hand.
# (x + 1)(x + 2) = x^2 + 3x + 2  -> coeffs (2, 3, 1)
# (2x^2 + 1)(x + 3) = 2x^3 + 6x^2 + x + 3 -> coeffs (3, 1, 6, 2)
MULT_CASES = [
    ((1, 1), (2, 1), (2, 3, 1)),
    ((1, 0, 2), (3, 1), (3, 1, 6, 2)),
    ((1,), (5, 4, 3), (5, 4, 3)),        # constant times polynomial
    ((1, -1), (1, 1), (1, 0, -1)),       # (1 - x)(1 + x) = 1 - x^2
]


@pytest.mark.parametrize("method", ["convolution", "karatsuba", "fft"])
@pytest.mark.parametrize("a, b, expected", MULT_CASES)
def test_multiply_methods_correct(method, a, b, expected):
    """Each forced algorithm produces the correct product on known inputs."""
    p = poly(a)
    q = poly(b)
    result = p.multiply(q, method=method)
    # Compare coefficient-wise with tolerance, since fft returns floats.
    assert result.coeffs == pytest.approx(expected)


@pytest.mark.parametrize(
    "a, b",
    [
        ((1, 2, 3, 4), (5, 6, 7)),
        ((0, 0, 1), (1, 1, 1, 1)),
        ((2, -3, 0, 1, 5), (1, 0, -2)),
    ],
)
def test_multiply_methods_agree(a, b):
    """convolution, karatsuba and fft give the same product."""
    p = poly(a)
    q = poly(b)
    conv = p.multiply(q, method="convolution")
    karat = p.multiply(q, method="karatsuba")
    fft = p.multiply(q, method="fft")
    assert karat.coeffs == pytest.approx(conv.coeffs)
    assert fft.coeffs == pytest.approx(conv.coeffs)


def test_multiply_invalid_method_raises():
    """An unknown method string is rejected."""
    p = poly((1, 1))
    q = poly((1, 1))
    with pytest.raises(ValueError):
        p.multiply(q, method="nonsense")


def test_multiply_non_polynomial_raises():
    """Multiplying by a non-polynomial raises TypeError."""
    p = poly((1, 1))
    with pytest.raises(TypeError):
        p.multiply("hello", method="convolution")
