"""
ChatGPT helped write parts of this function/file.
"""

# region imports
# no special imports needed
# endregion


# region function definitions
def ode_system(x, Y):
    """
    Converts the second-order ODE

        y'' - y = x

    into a first-order system by letting:
        y1 = y
        y2 = y'

    Then:
        y1' = y2
        y2' = y + x

    :param x: current x value
    :param Y: list [y, y']
    :return: list [y', y'']
    """
    y = Y[0]
    yp = Y[1]

    dy_dx = yp
    dyp_dx = y + x

    return [dy_dx, dyp_dx]


def improved_euler_step(x, Y, h):
    """
    Performs one Improved Euler (Heun) step for the system.

    :param x: current x value
    :param Y: current state [y, y']
    :param h: step size
    :return: updated state [y, y']
    """
    f1 = ode_system(x, Y)

    Y_predict = [
        Y[0] + h * f1[0],
        Y[1] + h * f1[1]
    ]

    f2 = ode_system(x + h, Y_predict)

    Y_next = [
        Y[0] + (h / 2.0) * (f1[0] + f2[0]),
        Y[1] + (h / 2.0) * (f1[1] + f2[1])
    ]

    return Y_next


def rk4_step(x, Y, h):
    """
    Performs one 4th-order Runge-Kutta step for the system.

    :param x: current x value
    :param Y: current state [y, y']
    :param h: step size
    :return: updated state [y, y']
    """
    k1 = ode_system(x, Y)

    Y2 = [
        Y[0] + (h / 2.0) * k1[0],
        Y[1] + (h / 2.0) * k1[1]
    ]
    k2 = ode_system(x + h / 2.0, Y2)

    Y3 = [
        Y[0] + (h / 2.0) * k2[0],
        Y[1] + (h / 2.0) * k2[1]
    ]
    k3 = ode_system(x + h / 2.0, Y3)

    Y4 = [
        Y[0] + h * k3[0],
        Y[1] + h * k3[1]
    ]
    k4 = ode_system(x + h, Y4)

    Y_next = [
        Y[0] + (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        Y[1] + (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    ]

    return Y_next


def improved_euler_method(y0, yp0, h, x_final):
    """
    Solves the IVP using the Improved Euler method.

    :param y0: initial value y(0)
    :param yp0: initial value y'(0)
    :param h: step size
    :param x_final: x value where solution is desired
    :return: [y, y'] at x_final
    """
    x = 0.0
    Y = [y0, yp0]

    while x < x_final - 1e-12:
        step = min(h, x_final - x)
        Y = improved_euler_step(x, Y, step)
        x += step

    return Y


def rk4_method(y0, yp0, h, x_final):
    """
    Solves the IVP using the 4th-order Runge-Kutta method.

    :param y0: initial value y(0)
    :param yp0: initial value y'(0)
    :param h: step size
    :param x_final: x value where solution is desired
    :return: [y, y'] at x_final
    """
    x = 0.0
    Y = [y0, yp0]

    while x < x_final - 1e-12:
        step = min(h, x_final - x)
        Y = rk4_step(x, Y, step)
        x += step

    return Y
# endregion


# region main
def main():
    """
    Solves the initial value problem

        y'' - y = x

    using both the Improved Euler method and the Runge-Kutta method.
    The user provides:
        y(0)
        y'(0)
        step size
        x value where y and y' are desired

    The program then reports y and y' from both methods and gives
    the user the option to compute again.
    """
    go_again = True

    while go_again:
        print("For the initial value problem y'' - y = x")

        y0 = float(input("Enter the value of y at x=0: "))
        yp0 = float(input("Enter the value of y' at x=0: "))
        h = float(input("Enter the step size for the numerical solution: "))
        x_final = float(input("At what value of x do you want to know y and y'? "))

        IE = improved_euler_method(y0, yp0, h, x_final)
        RK = rk4_method(y0, yp0, h, x_final)

        print(f"\nAt x={x_final:.3f}")
        print(f"For the improved Euler method: y={IE[0]:.3f}, and y'={IE[1]:.3f}")
        print(f"For the Runge-Kutta method: y={RK[0]:.3f}, and y'={RK[1]:.3f}")

        ans = input("\nDo you want to compute at a different x? (Y/N) ").strip().lower()
        go_again = ans.startswith('y')
        print()


if __name__ == "__main__":
    main()
# endregion