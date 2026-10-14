from numpy import linspace as ls
from numpy import pi
from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import os

class solver:
	def __init__(self, p0, N=10, dt=0.01, T=1):
		self.T = T
		self.M = 2 * N + 1
		self.grid = ls(-pi, pi, self.M, endpoint=False)
		self.dx = 2 * pi / self.M
		self.dt = dt
		self.nsteps = int(T / dt)
		self.p = p0(self.grid)
		self.ks = fftfreq(self.M, d=self.dx) * 2 * pi
		self.t = 0
		self.snapshots = {0.0: self.p.copy()}

	def store(self):
		self.snapshots[round(self.t, 1)] = self.p.copy()

	def step(self):
		F_p = fft(self.p)
		F_dp = 1j * self.ks * F_p
		dp_dx = ifft(F_dp).real
		self.p -= self.dt * self.p * dp_dx
		self.t += self.dt

def plot(solver, filename, folder='Figures'):
	os.makedirs(folder, exist_ok=True)
	#filepath = os.path.join(folder, filename)

	plt.figure(figsize=(9, 5))
	for t, values in solver.snapshots.items():
		plt.plot(solver.grid, values, label=f't = {t:.1f}')
	plt.xlabel('x')
	plt.ylabel('p(x,t)')
	plt.title('Spectral Approximation of Solution to the 1D Advection Equation')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(folder, filename))
	plt.show()
	plt.close()

















































































