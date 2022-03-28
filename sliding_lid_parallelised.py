#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:38:04 2021

@author: pablo
"""
from mpi4py import MPI
from mpi_helper import save_mpiio
import numpy as np
import sys
import os

path = 'vel_fields'
if not os.path.exists(path):
    os.makedirs(path)

###############################################################################
# ADJUSTABLE PARAMETERS

Ny = 300 # height of the grid
Nx = 300 # width of the grid

# subdivision of the grid y- and x-direction
# MUST correspond to the number of allocated processors
# MUST be >= 2
y_pgrid = 6
x_pgrid = 6

# current version: only movement of upper wall supported
#          LEFT(x=0)  BOTTOM(y=0) RIGHT  UPPER
W_SPEED =  [0.0,       0.0,       0.0,   0.1] # movement-speeds of the walls
#direction: +y (up)   +x (right)   +y      +x

# 0 < omega <= 1.7 (entsprichst viscosity = 1/3 * (1/omega - 1/2))
# smaller omega -> fluid is thicker
# omega = 0.891 (Wasser)
OMEGA = 1.7

NSTEPS = 100000
SAVEPOINTS = [600, 1000, 3000, 5000, 10000, 300000, 50000, 70000, 100000]

###############################################################################
###############################################################################

# number of Channels
Nch = 9

# velocity sets
c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],
              [0,  0,  1,  0, -1,  1,  1, -1, -1]])

# weight set
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# diverse Variablen für MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # eigene ID des aktuellen Prozesses
size = comm.Get_size() # Gesamtanzahl an Prozessen
if size < 4:
    print("Number of processors must be at least 4!")
    sys.exit()
if rank == 0 and size != y_pgrid * x_pgrid:
    print(f"Anzahl Prozessoren ({size}) != y_pgrid({y_pgrid}) * x_pgrid({x_pgrid}). Eines von beiden anpassen!")
    sys.exit()
# subdivision of the sub-grids
# i.e. with x_pgrid = y_pgrid = 3
#   (Rank 6) (Rank 7) (Rank 8)
#   (Rank 3) (Rank 4) (Rank 5)
#   (Rank 0) (Rank 1) (Rank 2)
cartcomm = comm.Create_cart((y_pgrid, x_pgrid))
# wenn Rückgabewerte <= -1: Gitterbereich befindet sich am entsprechenden Rand des globalen Gitters
# Tupel (source_xxx, dest_xxx)
sd_right = cartcomm.Shift(1, 1)
sd_upper = cartcomm.Shift(0, 1)
sd_left = cartcomm.Shift(1, -1)
sd_lower = cartcomm.Shift(0,-1)



###############################################################################
# INITIAL STATE
def init():
    
    # every process must know, if *and at which* boundary it lays
    WALLS = [False, False, False, False]
    if sd_right[1] <= -1:
        WALLS[2] = True
    if sd_upper[1] <= -1:
        WALLS[3] = True
    if sd_left[1] <= -1:
        WALLS[0] = True
    if sd_lower[1] <= -1:
        WALLS[1] = True

    # GLOBAL BOUNDARIES OF THE SUB-GRIDS
    # rechts
    xmax = int(round(Nx / x_pgrid * ((rank % x_pgrid) +1) ))
    # oben
    ymax = int(round(Ny / y_pgrid * (int(rank / x_pgrid) +1)))-1
    # links
    xmin = int(round(Nx / x_pgrid * (rank % x_pgrid) ))
    # unten
    ymin = int(round(Ny / y_pgrid * int(rank / x_pgrid) ))
    # info about the Sub-Grids wanted?? uncomment the following lines:
    # print(rank, WALLS, "\n", sd_left, "\n", sd_lower, "\n", sd_right, "\n", sd_upper, "\ny=[", ymin, ymax, "]  x=[", xmin, xmax,"]")
    # sys.exit()

    # INITIALISE f
    # size of sub-grid + 2 for ghost-nodes/layers
    f = np.ones((Nch, ymax-ymin+2, xmax-xmin+2))
    
    for channel in np.arange(9):
        f[channel] = w_i[channel]
    den_grid = density(f)
    vel_field = velocity(f)
    
    return den_grid, vel_field, f, ymin, ymax, xmin, xmax, WALLS



###############################################################################
# (1) STREAMING
def streaming_step(f, WALLS):
    # before performing streaming in Subgrid, get the right values for ghost-nodes
    left_sendbuf  = f[:, :, 1].copy()
    left_recvbuf  = f[:, :,-1].copy()
    comm.Sendrecv(left_sendbuf, sd_left[1], recvbuf=left_recvbuf, source=sd_left[0])
    if not WALLS[2]:
        f[:, :, -1] = left_recvbuf

    right_sendbuf = f[:, :,-2].copy()
    right_recvbuf = f[:, :, 0].copy()
    comm.Sendrecv(right_sendbuf, sd_right[1], recvbuf=right_recvbuf, source=sd_right[0])
    if not WALLS[0]:
        f[:, :, 0] = right_recvbuf

    lower_sendbuf = f[:, 1, :].copy()
    lower_recvbuf = f[:,-1, :].copy()
    comm.Sendrecv(lower_sendbuf, sd_lower[1], recvbuf=lower_recvbuf, source=sd_lower[0])
    if not WALLS[3]:
        f[:,-1, :] = lower_recvbuf
        
    upper_sendbuf = f[:,-2, :].copy()
    upper_recvbuf = f[:, 0, :].copy()
    comm.Sendrecv(upper_sendbuf, sd_upper[1], recvbuf=upper_recvbuf, source=sd_upper[0])
    if not WALLS[1]:
        f[:, 0, :] = upper_recvbuf

    for channel in np.arange(1,9):
        f[channel,:,:] = np.roll(f[channel,:,:], shift=c.T[channel],axis=(1,0))

    return f


###############################################################################
# (2) BOUNDARY CONDITION (BOUNCE BACK)
def apply_boundary(f, f_copy, WALLS):

    # rechte Wand
    if WALLS[2]:
        f[6, :, -1] = f_copy[8, :, -1]
        f[3, :, -1] = f_copy[1, :, -1]
        f[7, :, -1] = f_copy[5, :, -1]

    # obere Wand
    if WALLS[3]:
        den_wall = f[6, -1, :] + f[2, -1, :] + f[5, -1, :] + f[3, -1, :] + f[0, -1, :] + f[1, -1, :] + f_copy[6, -1, :] + f_copy[2, -1, :] + f_copy[5, -1, :]
        f[7, -1, :] = f_copy[5, -1, :] - 6 * w_i[7] * den_wall * W_SPEED[3]
        f[4, -1, :] = f_copy[2, -1, :]
        f[8, -1, :] = f_copy[6, -1, :] - (-6 * w_i[8] * den_wall * W_SPEED[3])

    # linke Wand
    if WALLS[0]:
        f[5, :, 0] = f_copy[7, :, 0]
        f[1, :, 0] = f_copy[3, :, 0]
        f[8, :, 0] = f_copy[6, :, 0]

    # untere Wand
    if WALLS[1]:
        f[6, 0, :] = f_copy[8, 0, :]
        f[2, 0, :] = f_copy[4, 0, :]
        f[5, 0, :] = f_copy[7, 0, :]

    return f

###############################################################################
# (3.1) MOMENTUM UPDATE
def density(f):
    '''
    RHO    calculates density of every gridpoint (= sum of velocities (0..9))
            IN:  f(v,y,x)
            OUT: density of each gridpoint (y, x)
    '''
    den_grid = np.einsum('ijk->jk',f)
    return den_grid


def velocity(f):
    '''
    v      calculates velocity-field for every gridpoint
            IN:  f(v,y,x)
            OUT: velocity field (divided by x- and y-component)
    '''
    vel_field = (1.0/density(f)) * np.einsum('in,ijk->njk', c.T, f)

    return vel_field


# (3.2) EQUILIBRIUM
def f_equilib(rho, vel, ymin, ymax, xmin, xmax):
    f_eq = np.zeros((Nch, ymax-ymin+2, xmax-xmin+2))

    _rd = np.einsum('njk,njk->jk', vel, vel)
    for channel in np.arange(9):
        _omrho = w_i[channel] * rho
        _st = np.einsum('n,njk->jk',c.T[channel],vel)
        _nd = np.einsum('jk,jk->jk', _st, _st)

        f_eq[channel] = _omrho * (1.0 + 3.0 * _st + 9/2 * _nd - 3/2 * _rd)
    return f_eq


# (3.3) RELAXATION
def relaxation(f, f_eq):
    '''
    Boltzmann transportation equation
    '''
    f_new = f + OMEGA * (f_eq - f)
    return f_new



###############################################################################
###############################################################################
def mass_check_simple(old_pdf, new_pdf):
    '''
    überprüft, ob Masse zwischen zwei Zeitschritten erhalten bleibt
    IN:  alte PDF, neue PDF
    OUT: True (wenn Masse erhalten geblieben)
    '''
    old_mass = np.sum(density(old_pdf[:, 1:-1, 1:-1]))
    new_mass = np.sum(density(new_pdf[:, 1:-1, 1:-1]))
    # Toleranz
    if np.abs(old_mass - new_mass) > 1e-08:
        print("#############################")
        print(f"final mass_diff = {old_mass-new_mass}")
        print("Simulation ist möglicherweise noch ausreichend genau...")
        return False
    return True




###############################################################################
###############################################################################

# INITIAL STATE
den_grid, vel_field, f, ymin, ymax, xmin, xmax, WALLS = init()

f_start = f.copy()


for step in np.arange(NSTEPS+1):
    old_f = f.copy()
# (1) STREAMING
    f = streaming_step(f, WALLS)
# (2) BOUNCE BACK
    f = apply_boundary(f, old_f, WALLS)
# (3.1) MOMENTUM UPDATE
    den_grid = density(f) # 1. rho
    vel_field = velocity(f) # local average velocity
# (3.2) EQUILIBRIUM    (ausgeglichenen Zustand des Systems berechnen)
    f_eq = f_equilib(den_grid, vel_field, ymin, ymax, xmin, xmax)
# (3.3) RELAXATION / COLLISION
    f = relaxation(f, f_eq)
    
    if step in SAVEPOINTS:
        save_mpiio(cartcomm, "vel_fields/ux%d_%d_%d.npy" % (step, Nx, Ny), vel_field[0, 1:-1, 1:-1])
        save_mpiio(cartcomm, "vel_fields/uy%d_%d_%d.npy" % (step, Nx, Ny), vel_field[1, 1:-1, 1:-1])


mass_check_simple(f_start, f)



