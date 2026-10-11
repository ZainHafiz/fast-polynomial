from fast_polynomial import Polynomial

if __name__ == "__main__":
    x = Polynomial((0, 1))
    p = x**5 + 5*x**2 + 1
    q = x**2 + 2*x + 1
    print("p(x) =", p)
    print("q(x) =", q)
    print("q(2) =", q(2))
    print("p + q =", p + q)
    print("p * q =", p * q)
    print("p' =", p.derivative())
    print("∫p dx =", p.integral())
    print("p^3 =", p**3)
    quotient, remainder = divmod(p, q)
    print(f"p/q = {quotient} r {remainder}")
