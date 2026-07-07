import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (7, 5)


def bisection(func, a, b, eps, max_iter=200):
    func_a, func_b = func(a), func(b)
    if func_a * func_b > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    rows = []
    n = 0
    func_c = 0
    c = 0
    width = 0

    while n <= max_iter:
        c = (a + b) / 2
        func_c = func(c)
        width = b - a
        n += 1
        rows.append((n, a, b, c, func_c))

        if abs(func_c) < eps or width < eps:
            return c, func_c, width, n, rows

        if func_a * func_c < 0:
            b, func_b = c, func_c
        else:
            a, func_a = c, func_c

    return c, func_c, width, n, rows


def print_table(rows):
    print(f"\n{'n':>3} {'a_n':>12} {'b_n':>12} {'c_n':>12} {'f(c_n)':>14}")
    for n, a, b, c, fc in rows:
        print(f"{n:>3} {a:>12.6f} {b:>12.6f} {c:>12.6f} {fc:>14.6f}")


def plot_function(f, a, b, root, title, filename):
    margin = 0.25 * (b - a)
    xs = np.linspace(a - margin, b + margin, 400)
    ys = [f(x) for x in xs]
    plt.figure()
    plt.axhline(0, color="black", linewidth=0.8)
    plt.plot(xs, ys, label="f(x)")
    plt.plot(root, f(root), "ro", label=f"root ≈ {root:.6f}")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def run_bisection_task(task_num = 0):
    labels = [
        "e^x - x^2 = 0",
        "x^3 - x - 2 = 0",
        "cos(x) - x = 0",
        "x^3 - 4x - 9 = 0",
        "e^-x - x = 0",
        "x^2 - 5 = 0",
    ]

    func = [
        lambda x: math.exp(x) - x ** 2,
        lambda x: x ** 3 - x - 2,
        lambda x: math.cos(x) - x,
        lambda x: x ** 3 - 4 * x - 9,
        lambda x: math.exp(-x) - x,
        lambda x: x ** 2 - 5,
    ][task_num]

    a, b = [
        [-2, 0],
        [1, 2],
        [0, 1],
        [2, 3],
        [0, 1],
        [2, 3],
    ][task_num]

    eps = [
        0.001,
        1e-5,
        1e-6,
        1e-5,
        1e-6,
        1e-5,
    ][task_num]

    label = labels[task_num]

    print("=" * 70)
    print(f"BISECTION METHOD: {label} on [{a}, {b}], eps = {eps}")
    print("=" * 70)

    # a) verify sign change
    func_a, func_b = func(a), func(b)
    print(f"\nf({a}) = {func_a:.6f}")
    print(f"f({b}) = {func_b:.6f}")
    if func_a * func_b < 0:
        print("Sign change confirmed: a root exists in the interval.")
    else:
        raise ValueError("No sign change on the given interval")

    # b)+c) run iterations and record the log
    root, f_root, width, n_iter, rows = bisection(func, a, b, eps)
    print_table(rows)

    # d) report final result
    print(f"\nApproximate root: {round(root, 4)}")
    print(f"f(root) = {f_root:.6f}")
    print(f"Total number of iterations: {n_iter}")

    plot_function(func, a, b, root, f"Bisection method: {label} on [{a}, {b}]", f"bisection_task_{task_num + 1}.png")


if __name__ == "__main__":
    user_input = input("Enter task number (0-5): ").strip()
    run_bisection_task(int(user_input) if user_input else 0)
