#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 15:53:17 2021

@author: pablo
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

Nx = 300
Ny = 300
W_VEL = 0.1

meshX,meshY = np.meshgrid(np.arange(Nx),np.arange(Ny)) # Gitter zum Plotten


for step in np.arange(100001):
    try:
        ux = np.load("vel_fields/ux%d_%d_%d.npy" % (step, Nx, Ny))
        uy = np.load("vel_fields/uy%d_%d_%d.npy" % (step, Nx, Ny))
        if step != 1:
            fig = plt.figure()
            ax = fig.add_subplot()
            # ax.quiver(meshX[ymin:ymax,xmin:xmax],
            #           meshY[ymin:ymax,xmin:xmax],
            #           vel_field[0,ymin:ymax,xmin:xmax],
            #           vel_field[1,ymin:ymax,xmin:xmax])
            if (ux==0).all() and (uy==0).all():
                print("alle null in Schritt", step)
            else:
                strm = ax.streamplot(meshX, meshY, ux, uy,
                                     color=np.sqrt(ux**2 + uy**2), linewidth=2,
                                     cmap='viridis_r',
                                     norm=colors.Normalize(vmin=0, vmax=W_VEL))
                fig.colorbar(strm.lines)
                # quiv = ax.quiver(meshX, meshY, vel_field[0], vel_field[1])
            plt.xlim(0-Nx/50, Nx+Nx/50)
            plt.ylim(0-Nx/50, Ny+Nx/50)
            plt.hlines([0, Ny-1], [0, 0], [Nx-1, Nx-1],
                       color=['k',"red"], linewidth=0.8)
            plt.vlines([0, Nx-1], [0, 0], [Ny-1, Ny-1],
                       color='k', linewidth=0.8)
            ax.set_title(f"{step}. timestep")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            plt.savefig("%d.pdf" % step)
            plt.show()
    except OSError:
        pass



###############################################################################
###############################################################################
# PLOTTING RESULTS OF MEASURED RUNTIMES
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# max CPUs and resulting number of nodes used
nodes_1 = 36
nodes_2 = 64
nodes_3 = 81
nodes_4 = 144
nodes_7 = 196

###############################################################################
# laptop time (50 minutes runtime -> mlups=3)
plt.scatter(1, 3, c="k")

###############################################################################
# results from cluster with 300x300-Grid and 100000 Steps
Nx = Ny = 300
NSTEPS = 100000
# number of allocated processors
npr_300 = np.array([4, 6, 8, 9, 12, 16, 20, 25, 36, 42, 64, 72, 81, 100, 144])
# runtime in seconds
t_300 = np.array([485, 387, 379, 342, 255, 144, 155, 152, 112, 124, 109, 107, 103, 99, 100])
mlups_300 = (Nx*Ny*NSTEPS/1000000)/t_300

plt.plot(npr_300, mlups_300, linestyle='dashed', label="300x300", c="black")

upto_n1 = 15*[False]
upto_n2 = 15*[False]
upto_n3 = 15*[False]
upto_n4 = 15*[False]
upto_n7 = 15*[False]
for i, elem in enumerate(npr_300):
    if elem <= nodes_1:
        upto_n1[i] = True
    elif elem <= nodes_2:
        upto_n2[i] = True
    elif elem <= nodes_3:
        upto_n3[i] = True
    elif elem <= nodes_4:
        upto_n4[i] = True
    else:
        upto_n7[i] = True
plt.scatter(npr_300[upto_n1], mlups_300[upto_n1], c="red", label="1 node")
plt.scatter(npr_300[upto_n2], mlups_300[upto_n2], c="yellowgreen", label="2 nodes")
plt.scatter(npr_300[upto_n3], mlups_300[upto_n3], c="violet", label="3 nodes")
plt.scatter(npr_300[upto_n4], mlups_300[upto_n4], c="cyan", label="4 nodes")

###############################################################################
# results from cluster with 1000x1000-Grid and 100000 Steps
Nx = Ny = 1000
npr_1000 = np.array([36, 64, 100, 144, 196])
t_1000 = np.array([1415, 657, 408, 258, 222])
mlups_1000 = (Nx*Ny*NSTEPS/1000000)/t_1000

plt.plot(npr_1000, mlups_1000, linestyle='dashed', label="1000x1000", c="gray")

upto_n1 = 5*[False]
upto_n2 = 5*[False]
upto_n3 = 5*[False]
upto_n4 = 5*[False]
upto_n7 = 5*[False]
for i, elem in enumerate(npr_1000):
    if elem <= nodes_1:
        upto_n1[i] = True
    elif elem <= nodes_2:
        upto_n2[i] = True
    elif elem <= nodes_4:
        upto_n4[i] = True
    else:
        upto_n7[i] = True
plt.scatter(npr_1000[upto_n1], mlups_1000[upto_n1], c="red")
plt.scatter(npr_1000[upto_n2], mlups_1000[upto_n2], c="yellowgreen")
plt.scatter(npr_1000[upto_n4], mlups_1000[upto_n4], c="cyan")
plt.scatter(npr_1000[upto_n7], mlups_1000[upto_n7], c="darkred", label="7 nodes")

###############################################################################
plt.xscale('log')
plt.yscale('log')
plt.xlabel("number of processors")
plt.ylabel("MLUPS")
plt.legend()
ax.grid(which='both')
ax.grid(which='minor', alpha=0.4, linestyle='--')
plt.savefig("mlups.pdf")
plt.show()
