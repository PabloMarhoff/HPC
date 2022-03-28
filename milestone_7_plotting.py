#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 15:53:17 2021

@author: pablo
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

############################
Nx = 300
Ny = 300
W_SPEED = 0.1
NSTEPS = 100000
############################

meshX,meshY = np.meshgrid(np.arange(Nx),np.arange(Ny)) # Gitter zum Plotten


for step in np.arange(NSTEPS+1):
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
                                     norm=colors.Normalize(vmin=0, vmax=W_SPEED))
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



