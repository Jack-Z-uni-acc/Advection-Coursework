import spectralTools as st
from numpy import sin, isclose
from advectionHelpers import cosBell_01_05, indicator_01_05

cos_solver = st.solver(cosBell_01_05)
square_solver = st.solver(indicator_01_05)

for n in range(1, cos_solver.nsteps + 1):
	cos_solver.step()
	square_solver.step()
	if isclose(cos_solver.t, cos_solver.T / 2) or isclose(cos_solver.t, cos_solver.T):
		cos_solver.store()
		square_solver.store()

st.plot(cos_solver, '1D-advection-solution.png')
st.plot(square_solver, '1D-advection-solution_square.png')