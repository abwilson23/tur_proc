import matplotlib.pyplot as plt
import numpy as np

from tutils import BaseStateSystem

#######################################################
# Laplacian for reaction equation updates
#######################################################

def laplacian1D(a, dx):
    return (
        - 2 * a
        + np.roll(a,1,axis=0) 
        + np.roll(a,-1,axis=0)
    ) / (dx ** 2)

def random_initialiser(shape):
    return(
        np.random.normal(loc=0, scale=0.05, size=shape),
        np.random.normal(loc=0, scale=0.05, size=shape)
    )

def function_initialiser(shape,f1,f2):
    space = range(shape)
    return(
        np.fromiter(map(f1,space),float),
        np.fromiter(map(f2,space),float)
    )

#######################################################
# Initialisation, eq_point, and perturbation choices
#######################################################

# Initialisation functions
f1 = lambda x : eq_pt*(1 + pert_a(x))
f2 = lambda x : eq_pt*(1 + pert_b(x))

# Perturbation from equilibrium
def pert_a(x): return np.cos(14*np.pi*x/100)
def pert_b(x): return np.cos(14*np.pi*x/100)

# Reaction equations
def Ra(a,b): return a - a ** 3 - b + alpha
def Rb(a,b): return (a - b) * beta

# Diffusion and reaction constants 
Da, Db, alpha, beta = 1, 20.3, -0.005, 10

# Computing the equilibrium point of the system. 
eq_pt = np.cbrt(alpha)

#######################################################
# Main class #
#######################################################

class OneDimensionalRDEquations(BaseStateSystem):
    def __init__(self, Da, Db, Ra, Rb,
                 initialiser=random_initialiser,
                 width=1000, dx=1, 
                 dt=0.1, steps=1):
        
        self.Da = Da
        self.Db = Db
        self.Ra = Ra
        self.Rb = Rb
        
        self.initialiser = initialiser
        self.width = width
        self.dx = dx
        self.dt = dt
        self.steps = steps
 
    def initialise(self):
        self.t = 0
        self.a, self.b = function_initialiser(self.width,f1,f2)
        
    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()

    def _update(self):
        # unpack so we don't have to keep writing "self"
        a,b,Da,Db,Ra,Rb,dt,dx = (
            self.a, self.b,
            self.Da, self.Db,
            self.Ra, self.Rb,
            self.dt, self.dx
        )

        La = laplacian1D(a, dx)
        Lb = laplacian1D(b, dx)
        
        delta_a = dt * (Da * La + Ra(a,b))
        delta_b = dt * (Db * Lb + Rb(a,b))

        self.a += delta_a
        self.b += delta_b
        
    def draw(self, ax):
        ax.clear()
        ax.plot(self.a, color="r", label="A")
        ax.plot(self.b, color="b", label="B")
        ax.legend()
        ax.set_ylim(-1,1)
        ax.set_title("t = {:.2f}".format(self.t))

    def custom_draw(self, vals, title, filename):
        fig, ax = self.initialise_figure()
        ax.clear()
        ax.plot(vals, color="r", label="A")
        ax.legend()
        ax.set_ylim(min(vals), max(vals))
        fig.savefig('plots/' + filename + '.png')
        plt.close()
    
    # plots the amplitude of a(x,t) over n_steps as a function of t by taking
    # the ptp (peak to peak) value of a(x,t) at each time t and plotting the result. 
    def plot_amplitude(self, n_steps):
        self.initialise() # reset equation before plotting
        amp = np.array([])

        for _ in range(n_steps):
            amp = np.append(amp, self.a.ptp()) # measure peak to peak, i.e., get amplitude
            self.update()

        title = "amp,t={0:.2f},n={1}".format(self.t, n_steps)
        filename = "amp_plot_n={}".format(n_steps)
        self.custom_draw(amp, title, filename)

    # tracks a(pt, t) for n_steps of time, t. Plots a as a function of t.
    def track_growth(self, pt, n_steps):
        self.initialise() # reset equation before tracking
        orbit_a = np.array([])

        for _ in range(n_steps):
            orbit_a = np.append(orbit_a, self.get_val(pt))
            self.update()

        # plotting the diffs
        #diffs = [orbit_a[i+1] - orbit_a[i] for i in range(0, n_steps-1)]
        #self.custom_draw(np.array(diffs), "diffs", "diffs")
        
        title = "growth,t={0:.2f},n={1},pt={2}".format(self.t, n_steps, pt)
        filename = "growth_x={0:.1f}_n={1}".format(pt, n_steps)
        self.custom_draw(orbit_a, title, filename)

    def get_val(self, pt):
        assert 0 <= pt < self.width, "Point is not in the domain"
        if pt in range(self.width):
            return self.a[pt]
        else: 
            return self.interpolate(pt)
        
    def interpolate(self, pt):
        x_vals = [int(np.floor(pt)), int(np.ceil(pt))]
        a_vals = [self.a[j] for j in x_vals]
        return np.interp(pt, x_vals, a_vals)


# Local playground #
width = 100
dx = 1
dt = 0.001

equation = OneDimensionalRDEquations(
    Da, Db, Ra, Rb, 
    width=width, dx=dx, dt=dt, 
    steps=100
)

# Pre-initialise our equation so we don't have to keep calling this in playground
equation.initialise()

#Equation.plot_time_evolution("1dRD_check_2.gif", n_steps=100)
#Equation.plot_evolution_outcome("1dRD.png", n_steps=100)
