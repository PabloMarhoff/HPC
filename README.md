## HPC
##### High-Performance Computing: Fluid Mechanics with Python

The main file of the project is called `milestone_7.py`. It contains a parallelised version of the lid-driven cavity simulation. The default values are:
- `Nx = Ny = 300` (size of the grid in x- and y-direction)  
- `x_pgrid = y_pgrid = 6` (number of subgrids in x- and y-direction; (x_pgrid\*y_pgrid) **must** be the same as the total number of allocated processors)
- `W_SPEED = 0.1` (velocity of the moving wall)
- `OMEGA = 1.7` (relaxation parameter)
- `NSTEPS = 100000` (number of to perform in the simulation)
- `SAVEPOINTS = [...]` (steps of the simulation to save as pdf-file)

If other values are wished, they can be adapted in the code. They can be found at the very beginning of the file.
  
After running the code once, `milestone_7_plotting.py` can be used to plot the pdf-files saved before. If `Nx`, `Ny`, `W_SPEED` or `NSTEPS` were changed for the simulation they **must** be changed here as well!!
