"""
ChatGPT helped write parts of this function/file.
"""

# region imports
from scipy.integrate import solve_ivp
from math import sin
import numpy as np
from matplotlib import pyplot as plt
# endregion


# region class definitions
class circuit():
    def __init__(self, R=10, L=20, C=0.05, A=20, w=20, p=0):
        """
        Creates an RLC circuit object for the network shown in the exam.

        The circuit has:
        - an input voltage source v(t) = A*sin(w*t + p)
        - an inductor L in series with the source
        - a resistor R and capacitor C connected in parallel

        State variables used in this model:
        x1 = iL = current through the inductor
        x2 = vc = voltage across the capacitor

        :param R: resistance in ohms
        :param L: inductance in henries
        :param C: capacitance in farads
        :param A: source voltage amplitude
        :param w: source angular frequency in rad/s
        :param p: source phase angle in radians
        """
        # region attributes
        self.R = R
        self.L = L
        self.C = C
        self.A = A
        self.w = w
        self.p = p

        self.t = None
        self.iL = None
        self.i1 = None
        self.i2 = None
        self.vc = None
        self.sln = None
        # endregion

    # region methods
    def v(self, t):
        """
        Computes the source voltage at time t.

        :param t: time in seconds
        :return: source voltage v(t)
        """
        return self.A * sin(self.w * t + self.p)

    def ode_system(self, t, X):
        """
        Defines the state equations for the circuit.

        Let:
        x1 = iL = inductor current
        x2 = vc = capacitor voltage

        Then:
        diL/dt = (v(t) - vc) / L
        dvc/dt = (iL - vc/R) / C

        since:
        iL = i1 + i2
        i1 = vc/R
        i2 = C dvc/dt

        :param t: the current time
        :param X: the current values of the state variables [iL, vc]
        :return: list of derivatives [diL/dt, dvc/dt]
        """
        iL = X[0]
        vc = X[1]

        diLdt = (self.v(t) - vc) / self.L
        dvcdt = (iL - vc / self.R) / self.C

        return [diLdt, dvcdt]

    def simulate(self, t=10, pts=500):
        """
        Simulates the transient response of the circuit.

        The initial conditions are assumed to be:
        iL(0) = 0
        vc(0) = 0

        :param t: time duration of the simulation in seconds
        :param pts: number of time points in the simulation
        :return: nothing, stores results as object attributes
        """
        self.t = np.linspace(0, t, pts)
        X0 = [0.0, 0.0]

        self.sln = solve_ivp(self.ode_system, [0, t], X0, t_eval=self.t)

        self.iL = self.sln.y[0]
        self.vc = self.sln.y[1]

        self.i1 = self.vc / self.R
        self.i2 = self.iL - self.i1

    def doPlot(self, ax=None):
        """
        Plots i1(t), i2(t), and vc(t) with currents on the left axis
        and capacitor voltage on the right axis.

        :param ax: optional axes object for GUI plotting
        :return: nothing
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            QTPlotting = False
        else:
            QTPlotting = True

        ax2 = ax.twinx()

        # currents on left axis
        ax.plot(self.t, self.i1, '-', color='black', label=r'$i_1(t)$')
        ax.plot(self.t, self.i2, '--', color='black', label=r'$i_2(t)$')

        # capacitor voltage on right axis
        ax2.plot(self.t, self.vc, ':', color='black', label=r'$v_c(t)$')

        # axis labels
        ax.set_xlabel('t (s)')
        ax.set_ylabel(r'$i_1,i_2(A)$')
        ax2.set_ylabel(r'$v_c(t)(V)$')

        # axis limits to resemble the example plot
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.06, 0.10)
        ax2.set_ylim(-0.50, 0.10)

        # grid
        ax.grid(True)

        # legends
        ax.legend(loc='upper right')
        ax2.legend(loc='lower right')

        if not QTPlotting:
            plt.show()
    # endregion


# endregion

# region function definitions
def main():
    """
    Solves Problem 3 from the exam.

    The program:
    1. Creates a circuit object with default values
    2. Solicits user input for R, L, C, amplitude, frequency, and phase
    3. Simulates the response for 10 seconds
    4. Plots i1, i2, and vc
    5. Allows the user to change parameters and simulate again
    """
    goAgain = True

    while goAgain:
        print("RLC Circuit Simulation\n")
        print("Press Enter to use the default values.\n")

        R_default = 10.0
        L_default = 20.0
        C_default = 0.05
        A_default = 20.0
        w_default = 20.0
        p_default = 0.0

        stR = input(f"Enter resistance R in ohms ({R_default}): ").strip()
        R = R_default if stR == "" else float(stR)

        stL = input(f"Enter inductance L in henries ({L_default}): ").strip()
        L = L_default if stL == "" else float(stL)

        stC = input(f"Enter capacitance C in farads ({C_default}): ").strip()
        C = C_default if stC == "" else float(stC)

        stA = input(f"Enter source amplitude A ({A_default}): ").strip()
        A = A_default if stA == "" else float(stA)

        stw = input(f"Enter source angular frequency w in rad/s ({w_default}): ").strip()
        w = w_default if stw == "" else float(stw)

        stp = input(f"Enter source phase p in radians ({p_default}): ").strip()
        p = p_default if stp == "" else float(stp)

        Circuit = circuit(R=R, L=L, C=C, A=A, w=w, p=p)

        Circuit.simulate(t=10, pts=1000)
        Circuit.doPlot()

        ans = input("\nDo you want to simulate again with different parameters? (Y/N): ").strip().lower()
        goAgain = ans.startswith('y')
# endregion


# region function calls
if __name__ == "__main__":
    main()
# endregion