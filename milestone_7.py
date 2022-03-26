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


Nch = 9 # Anzahl Channel
Ny = 300 # Höhe des Gitters
Nx = 300 # Breite des Gitters

# Anzahl Prozessoren in y- bzw. x-Richtung
# beide müssen >= 2 sein
y_pgrid = 6
x_pgrid = 6



# current version: only movement of upper wall supported
#          LEFT(x=0)  BOTTOM(y=0) RIGHT  UPPER
W_SPEED =  [0.0,       0.0,       0.0,   0.15] # movement-speeds of the walls
#direction: +y (up)   +x (right)   +y      +x


# velocity sets
c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],
              [0,  0,  1,  0, -1,  1,  1, -1, -1]])

# weight set
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# 0 < omega < 2 ("omega nicht größer als 1,7!") entsprichst Viskosität = 1/3 * (1/omega - 1/2)
# d.h. je kleiner omega, desto dickflüssiger
# omega = 0.891 (Wasser)
OMEGA = 1.7

###############################################################################
# diverse Variablen für MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # eigene ID des aktuellen Prozesses
size = comm.Get_size() # Gesamtanzahl an Prozessen
if size < 4:
    print("Größere Anzahl an Prozessen wählen (mind.4) ")
    sys.exit()
if rank == 0 and size != y_pgrid * x_pgrid:
    print("Anzahl Prozessoren ({size}) != y_pgrid({y_pgrid}) * x_pgrid({x_pgrid}). Eines von beiden anpassen!")
    sys.exit()
# Aufteilung des Gitters auf die Prozessoren
# z. B. bei 3 x 3
#   (Rank 6) (Rank 7) (Rank 8)
#   (Rank 3) (Rank 4) (Rank 5)
#   (Rank 0) (Rank 1) (Rank 2)
cartcomm = comm.Create_cart((y_pgrid, x_pgrid))
# wenn Rückgabewerte <= -1: Gitterbereich befindet sich am entsprechenden Rand des globalen Gitters
# Tupel (source_xxx, dest_xxx)
sd_left = cartcomm.Shift(1, -1)
sd_lower = cartcomm.Shift(0,-1)
sd_right = cartcomm.Shift(1, 1)
sd_upper = cartcomm.Shift(0, 1)
###############################################################################




def init():
    
    # Jeder Prozessor muss wissen, ob *und an welchem* Rand er sich befindet
    WALLS = [False, False, False, False]
    if sd_left[1] <= -1:
        WALLS[0] = True
    if sd_lower[1] <= -1:
        WALLS[1] = True
    if sd_right[1] <= -1:
        WALLS[2] = True
    if sd_upper[1] <= -1:
        WALLS[3] = True

    # GLOBALE GRENZEN DER LOKALEN GITTER
    # wenn Ny % y_pgrid == 0 bzw. Nx % x_pgrid == 0, werden Gitterpunkte möglichst gleichmäßig auf Prozessoren aufgeteilt
    # links
    xmin = int(round(Nx / x_pgrid * (rank % x_pgrid) ))
    # unten
    ymin = int(round(Ny / y_pgrid * int(rank / x_pgrid) ))
    # rechts
    xmax = int(round(Nx / x_pgrid * ((rank % x_pgrid) +1) ))
    # oben
    ymax = int(round(Ny / y_pgrid * (int(rank / x_pgrid) +1)))-1
    print(rank, WALLS, "\n", sd_left, "\n", sd_lower, "\n", sd_right, "\n", sd_upper, "\ny=[", ymin, ymax, "]  x=[", xmin, xmax,"]")
    sys.exit()

    # INITIALISIERE LOKALEN AUSSCHNITT VON f
    # Größe ergibt sich aus berechneten GLOBALEN GRENZEN + 2 für Ghost-Nodes/Wände
    f = np.ones((Nch, ymax-ymin+2, xmax-xmin+2))
    
    for channel in np.arange(9):
        f[channel] = w_i[channel]
    den_grid = density(f)                   # 1. rho
    vel_field = velocity(f)                 # 2. local average velocity
    
    return den_grid, vel_field, f, ymin, ymax, xmin, xmax, WALLS


###############################################################################
# (1) MOMENTUM UPDATE
def density(f):
    '''
    RHO    berechnet die Dichte von jedem Gitterpunkt (= Summe der Geschwindigkeiten (0..9))
            IN:  f(v,y,x)                    (GESCHWINDIGKEITEN, y, x)
            OUT: density of each gridpoint   (y, x)
    '''
    den_grid = np.einsum('ijk->jk',f)   # identisch zu np.sum(f, axis=0)
    return den_grid


def velocity(f):
    '''
    v      berechnet das Geschwindigkeitsfeld für die Gitterpunkte
            IN:  f(v,y,x)
            OUT: Geschwindigkeitsfeld (getrennt nach x- und y-Komponente)
    '''
    vel_field = (1.0/density(f)) * np.einsum('in,ijk->njk', c.T, f)

    return vel_field


###############################################################################
# (2) EQUILIBRIUM
def f_equilib(rho, vel, ymin, ymax, xmin, xmax):
    f_eq = np.zeros((Nch, ymax-ymin+2, xmax-xmin+2))

    _rd = np.einsum('njk,njk->jk', vel, vel) # LAUT FORMEL: ..., np.abs(vel), np.abs(vel))
    for channel in np.arange(9):
        _omrho = w_i[channel] * rho
        _st = np.einsum('n,njk->jk',c.T[channel],vel)
        _nd = np.einsum('jk,jk->jk', _st, _st) # _nd = np.square(_st)

        f_eq[channel] = _omrho * (1.0 + 3.0 * _st + 9/2 * _nd - 3/2 * _rd)
    return f_eq


###############################################################################
# (3) RELAXATION
def relaxation(f, f_eq):
    '''
    Boltzmann transportation equation
    '''
    # for channel in np.arange(9):
    f_new = f + OMEGA * (f_eq - f)
    return f_new


###############################################################################
# (4) STREAMING
def streaming_step(f, WALLS):
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
# (5) BOUNDARY CONDITION BOUNCE BACK
def apply_boundary(f, f_copy, WALLS):
    # linke Wand
    if WALLS[0]:
        f[5, :, 0] = f_copy[7, :, 0]
        f[1, :, 0] = f_copy[3, :, 0]
        f[8, :, 0] = f_copy[6, :, 0]

    # rechte Wand
    if WALLS[2]:
        f[6, :, -1] = f_copy[8, :, -1]
        f[3, :, -1] = f_copy[1, :, -1]
        f[7, :, -1] = f_copy[5, :, -1]

    # untere Wand
    if WALLS[1]:
        f[6, 0, :] = f_copy[8, 0, :]
        f[2, 0, :] = f_copy[4, 0, :]
        f[5, 0, :] = f_copy[7, 0, :]

    # obere Wand
    if WALLS[3]:
        den_wall = f[6, -1, :] + f[2, -1, :] + f[5, -1, :] + f[3, -1, :] + f[0, -1, :] + f[1, -1, :] + f_copy[6, -1, :] + f_copy[2, -1, :] + f_copy[5, -1, :]
        f[7, -1, :] = f_copy[5, -1, :] - 6 * w_i[7] * den_wall * W_SPEED[3]
        f[4, -1, :] = f_copy[2, -1, :]
        f[8, -1, :] = f_copy[6, -1, :] - (-6 * w_i[8] * den_wall * W_SPEED[3])

    return f




# (1) AUSGANGSZUSTAND
den_grid, vel_field, f, ymin, ymax, xmin, xmax, WALLS = init()

for step in np.arange(150001):
    old_f = f.copy()

# (2) STREAMING
    f = streaming_step(f, WALLS)

# (3) APPLY BOUNDARY
    f = apply_boundary(f, old_f, WALLS)

# (4.1) MOMENTUM UPDATE
    den_grid = density(f) # 1. rho
    vel_field = velocity(f) # local average velocity
# (4.2) EQUILIBRIUM    (ausgeglichenen Zustand des Systems berechnen)
    f_eq = f_equilib(den_grid, vel_field, ymin, ymax, xmin, xmax)
# (4.3) RELAXATION / COLLISION
    f = relaxation(f, f_eq)
    
    if step in [600, 1000, 3000, 5000, 10000, 300000, 50000, 70000, 150000]:
        save_mpiio(cartcomm, "vel_fields/ux%d_%d_%d.npy" % (step, Nx, Ny), vel_field[0, 1:-1, 1:-1])
        save_mpiio(cartcomm, "vel_fields/uy%d_%d_%d.npy" % (step, Nx, Ny), vel_field[1, 1:-1, 1:-1])
