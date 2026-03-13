"""
ChatGPT helped write parts of this function/file.
"""

# region imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.integrate import solve_ivp, quad
# endregion


# region function definitions
def S(x):
    """
    Numerically evaluates the Fresnel-type integral

        S(x) = integral from 0 to x of sin(t^2) dt

    using scipy.integrate.quad.

    :param x: upper limit of integration
    :return: value of S(x)
    """
    s = quad(lambda t: np.sin(t ** 2), 0, x)
    return s[0]


def Exact(x):
    """
    Computes the exact solution of the initial value problem.

    For
        y' = (y - 0.01x^2)^2 sin(x^2) + 0.02x
        y(0) = 0.4

    let z = y - 0.01x^2. Then z' = z^2 sin(x^2), which gives

        y(x) = 1 / (2.5 - S(x)) + 0.01x^2

    where
        S(x) = integral from 0 to x of sin(t^2) dt

    :param x: independent variable
    :return: exact y(x)
    """
    return 1.0 / (2.5 - S(x)) + 0.01 * x ** 2


def ODE_System(x, y):
    """
    Defines the first-order ODE system for solve_ivp.

    :param x: independent variable
    :param y: state vector, where y[0] is the dependent variable
    :return: list containing dy/dx
    """
    Y = y[0]
    Ydot = (Y - 0.01 * x ** 2) ** 2 * np.sin(x ** 2) + 0.02 * x
    return [Ydot]


def PlotResults(*args):
    """
    Produces the required plot for the exact and numerical solutions.

    :param args: xRange_Num, y_Num, xRange_Xct, y_Xct
    :return: nothing
    """
    xRange_Num, y_Num, xRange_Xct, y_Xct = args

    plt.figure(figsize=(8, 5))

    # exact solution as solid line
    plt.plot(xRange_Xct, y_Xct, '-', label='Exact')

    # numerical solution as upward-facing triangles
    plt.plot(xRange_Num, y_Num, '^', label='Numerical')

    # axis limits and labels
    plt.xlim(0.0, 6.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('x')
    plt.ylabel('y')

    # title
    plt.title("IVP: y' = (y - 0.01x^2)^2 sin(x^2) + 0.02x, y(0) = 0.4")

    # ticks inward on all sides
    plt.tick_params(axis='x', direction='in', top=True, bottom=True)
    plt.tick_params(axis='y', direction='in', left=True, right=True)

    # one-digit formatting on axes
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # legend
    plt.legend()

    plt.show()


def main():
    """
    This function solves the initial value problem for Problem 2.

        y' = (y - 0.01x^2)^2 sin(x^2) + 0.02x
        y(0) = 0.4

    It solves the problem numerically using solve_ivp over 0 <= x <= 5
    with step size h = 0.2, computes the exact solution using quad,
    and then plots both according to the exam formatting requirements.
    """
    xRange = np.arange(0.0, 5.0 + 0.2, 0.2)
    xRange_xct = np.linspace(0.0, 5.0, 500)
    Y0 = [0.4]

    sln = solve_ivp(ODE_System, [0, 5], Y0, t_eval=xRange)
    xctSln = np.array([Exact(x) for x in xRange_xct])

    PlotResults(xRange, sln.y[0], xRange_xct, xctSln)
# endregion


# region function calls
if __name__ == "__main__":
    main()
# endregion