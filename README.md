#################################################################
########################   PHY 571   ############################
#################################################################
-----------------------------------------------------------------
|             Bose Einstein condensates simulation              |
-----------------------------------------------------------------

Authors: Jonathan GAUTHIER
	 Maxime LOIL

This project is divided in two parts:
- 1D: simulation of 1D BECs
- 2D: simulation of 2D BECS

For both of them, simulation will be directed in the main.py file
while the other files contains useful functions or classes.

WARNING: In our environment, it was not possible to import other
=======	 files but to use the os.chdir method.
	 If you want to properly import the files in your 
	 environment you have to modify these files:
	 - 1D/production/animation.py
	 - 1D/production/g_variation.py
	 - 1D/production/main.py
	 - 2D/production/animation.py
	 - 2D/production/main.py

1D
****************************************

The main file is divided in 7 examples.
It shows how the external files can be used to simulate 1D BECs
and to analyse them.

List of external files:

- 1D/analysis/compute_energy.py
Give the energy of a state, and give the result of the 
hamiltonian over the state.

- 1D/design/analytical_solution.py
Construction of analytical solution of the quantum harmonic
oscillator with linear Schrödinger equation.

- 1D/design/crank_nicolson_1D.py
Frame to use the Crank-Nicolson algorithm and to propagate the
solution in real time or imaginary time.

- 1D/design/potential.py
Specific class to define potential and its parameters. The
function f is directly used in the Crank-Nicolson algorithm.

- 1D/production/animation.py
Evolve the solution with the equation of Gross-Pitaevskii. The
calculations are made during the ploting.

- 1D/production/g_variation.py
Specific file to calculate the energy of the BECs regarding the
Ng factor. Calculates also the standard deviation of the wave
function, often written as Δx, regarding the Ng factor.

- 1D/production/ground_storage.py
Run the simulation without plot to get the ground state.
Allow also to save and load ground states.

- 1D/production/main.py
The main file where the instructions are given.


2D
****************************************

The main file is divided in 4 examples.
It shows how the external files can be used to simulate 2D BECs
and to analyse them.

List of external files:

- 2D/analysis/compute_energy.py
Give the energy of a state, and give the result of the
hamiltonian over the state.

- 2D/design/analytical_solution.py
Construction of analytical solution of the quantum harmonic
oscillator with linear Schrödinger equation.

- 2D/design/crank_nicolson_2D.py
Frame to use the Crank-Nicolson algorithm and to propagate the
solution in real time or imaginary time.

- 2D/design/potential.py
Specific class to define potential and its parameters. The
function f is directly used in the Crank-Nicolson algorithm.
The double potential is also implemented. remove_barrier allows
the barrier between the two "holes" of the double potential to
vanish. It was coded to simulate matter interferences.

- 2D/production/animation.py
Evolve the solution with the equation of Gross-Pitaevskii. The
calculations are made during the ploting. The animation can be
saved in a MP4 file.

- 2D/production/ground_storage.py
Run the simulation without plot to get the ground state.
Allow also to save and load ground states.

- 2D/production/vortex.py
Specific class to create vortices.

- 2D/production/main.py
The main file where the instructions are given.