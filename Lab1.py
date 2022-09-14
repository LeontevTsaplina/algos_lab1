import random
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import progressbar
from statistics import mean
from typing import Callable


def cycle_function(function: Callable) -> None:
    """
    A function that plots the time complexity for a given function

    :param function: Given function
    :type function: Callable
    """
    labels = {
        "const_func": "f(v)=1",
        "sum_func": "f(v)=sum(v)",
        "prod_func": "f(v)=prod(v)",
        "polynom_func_direct": "P(x)=sum(vk(x^(k-1)))",
        "polynom_func_horner": "P(x)=vk+x(vk-1+x(...))",
        "bubble_sort": "bubble(v)",
        "quick_sort": "quick(v)",
        "timsort": "timsort(v)",
        "matrix_prod": "matrix_A*matrix_B"
    }

    if function.__name__ == 'matrix_prod':
        p = 201
    elif function.__name__ in ('polynom_func_direct', 'bubble_sort'):
        p = 701
    else:
        p = 2001

    times = []

    widgets = [
        labels[function.__name__],
        "   ",
        progressbar.Bar(),
        ' (', progressbar.Percentage(), ') ',
        ' [', progressbar.Counter(), '/', str(p - 1), '] ',
    ]

    with progressbar.ProgressBar(max_value=p-1, widgets=widgets) as bar:
        for n in range(1, p):
            current_times = []
            if function.__name__ != 'matrix_prod':
                vec = [random.randint(0, 1000) for _ in range(n)]
            else:
                matrix_a = [[random.randint(0, 1000) for _ in range(n)] for _ in range(n)]
                matrix_b = [[random.randint(0, 1000) for _ in range(n)] for _ in range(n)]

            for i in range(5):
                start_time = time.perf_counter()

                if function.__name__ not in ('polynom_func_direct', 'polynom_func_horner', 'matrix_prod'):
                    function(vec)
                elif function.__name__ != 'matrix_prod':
                    function(vec, 1.5)
                else:
                    function(matrix_a, matrix_b)

                end_time = time.perf_counter()
                current_times.append(end_time - start_time)

            times.append(mean(current_times))
            bar.update(n)

    x = np.linspace(1, p - 1, p - 1)
    y = times

    plt.plot(times, linewidth=0.5, label="Experimental")
    if function.__name__ == "const_func":
        approx_func = approx_const
    elif function.__name__ in ("sum_func", "prod_func", "polynom_func_horner"):
        approx_func = approx_linear
    elif function.__name__ in ("quick_sort", "timsort", "polynom_func_direct"):
        approx_func = approx_nlogn
    elif function.__name__ == "bubble_sort":
        approx_func = approx_square
    elif function.__name__ == "matrix_prod":
        approx_func = approx_cube

    popt, _ = curve_fit(approx_func, x, y, maxfev=100000)
    plt.plot(x, approx_func(x, *popt), linewidth=1, label="Theoretical")
    plt.ylim(0, max(times) + max(times) / 10)
    plt.title("{}".format(labels[function.__name__]))
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Time")
    plt.show()


def const_func(vector: list) -> int:
    """
    A function that returns 1 for given vector

    :param vector: Given vector
    :type vector: list
    :return: 1
    :rtype: int
    """
    return 1


def sum_func(vector: list) -> int:
    """
    A function that returns sum for elements of given vector

    :param vector: Given vector
    :type vector: list
    :return: sum for elements of given vector
    :rtype: int
    """
    result = 0

    for elem in vector:
        result += elem

    return result


def prod_func(vector: list) -> int:
    """
    A function that returns production for elements of given vector

    :param vector: Given vector
    :type vector: list
    :return: production for elements of given vector
    :rtype: int
    """
    result = 0

    for elem in vector:
        result *= elem

    return result


def polynom_func_direct(vector: list, x: float) -> float:
    """
    A function that returns direct calculation of polynom for elements of given vector


    :param vector: Given vector
    :param x: coefficient
    :type vector: list
    :type x: float
    :return: calculation of polynom for elements of given vector
    :rtype: float
    """
    return sum([(elem * x ** (index - 1)) for index, elem in enumerate(vector)])


def polynom_func_horner(vector: list, x: float) -> float:
    """
    A function that returns calculation of polynom for elements of given vector by Horner's method


    :param vector: Given vector
    :param x: coefficient
    :type vector: list
    :type x: float
    :return: calculation of polynom for elements of given vector
    :rtype: float
    """
    result = 0

    for elem in vector[::-1]:
        result += elem
        result *= x

    return result


def bubble_sort(vector: list) -> list:
    """
    A function that returns sorted vector of given vector by bubble sort


    :param vector: Given vector
    :type vector: list
    :return: sorted vector
    :rtype: list
    """
    result = vector

    swapped = True

    while swapped:
        swapped = False
        for i in range(len(result) - 1):
            if result[i] > result[i + 1]:
                result[i], result[i + 1] = result[i + 1], result[i]
                swapped = True

    return result


def quick_sort(vector: list) -> list:
    """
    A function that returns sorted vector of given vector by quick sort


    :param vector: Given vector
    :type vector: list
    :return: sorted vector
    :rtype: list
    """
    if len(vector) <= 1:
        return vector
    else:
        separator = random.choice(vector)
        left_nums = []
        mid_nums = []
        right_nums = []
        for elem in vector:
            if elem < separator:
                left_nums.append(elem)
            elif elem > separator:
                right_nums.append(elem)
            else:
                mid_nums.append(elem)
        return quick_sort(left_nums) + mid_nums + quick_sort(right_nums)


def timsort(vector: list) -> list:
    """
    A function that returns sorted vector of given vector by timsort


    :param vector: Given vector
    :type vector: list
    :return: sorted vector
    :rtype: list
    """
    return sorted(vector)


def matrix_prod(matrix_1: list, matrix_2: list) -> list:
    """
    A function that returns prod of 2 matrix

    :param matrix_1: matrix 1
    :param matrix_2: matrix 2
    :type matrix_1: list
    :type matrix_2: list
    :return: prod of 2 matrix
    :rtype: list
    """
    result = []
    for i in range(len(matrix_1)):

        row = []
        for j in range(len(matrix_2[0])):

            product = 0
            for k in range(len(matrix_1[i])):
                product += matrix_1[i][k] * matrix_2[k][j]
            row.append(product)

        result.append(row)

    return result


def approx_const(x, a):
    """
    A function that returns result of constant function

    :param x: param x
    :param a: coefficient a
    :return: result of function
    """
    return 0 * x + a


def approx_linear(x, a, b):
    """
    A function that returns result of linear function

    :param x: param x
    :param a: coefficient a
    :param b: coefficient b
    :return: result of function
    """

    return a * x + b


def approx_nlogn(x, a, b, c):
    """
    A function that returns result of linear-logarithmic function

    :param x: param x
    :param a: coefficient a
    :param b: coefficient b
    :param c: coefficient c
    :return: result of function
    """

    return a * x * np.log(x * b) + c


def approx_square(x, a, b, c):
    """
    A function that returns result of square function

    :param x: param x
    :param a: coefficient a
    :param b: coefficient b
    :param c: coefficient c
    :return: result of function
    """

    return a * x ** 2 + b * x + c


def approx_cube(x, a, b, c, d):
    """
    A function that returns result of cube function

    :param x: param x
    :param a: coefficient a
    :param b: coefficient b
    :param c: coefficient c
    :param d: coefficient d
    :return: result of function
    """

    return a * x ** 3 + b * x ** 2 + c * x + d


if __name__ == '__main__':
    cycle_function(const_func)  # const function call
    cycle_function(sum_func)  # sum function call
    cycle_function(prod_func)  # prod function call
    cycle_function(polynom_func_direct)  # polynom direct calc function call
    cycle_function(polynom_func_horner)  # polynom horner's method function call
    cycle_function(bubble_sort)  # vectors bubble sort
    cycle_function(quick_sort)  # vectors quick sort
    cycle_function(timsort)  # vectors timsort
    cycle_function(matrix_prod)  # matrix prod
