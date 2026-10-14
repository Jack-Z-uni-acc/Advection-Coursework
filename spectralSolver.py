import spectralTools as st
from numpy import sin, isclose

def p0(x):
	return sin(x)

solver = st.solver(p0)

for n in range(1, solver.nsteps + 1):
	solver.step()
	if isclose(solver.t, solver.T / 2) or isclose(solver.t, solver.T):
		solver.store()

st.plot(solver, '1D-advection-solution.png')