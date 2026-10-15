# Numerical schemes for solving the advection equation
# Authors: Hilary Weller <h.weller@reading.ac.uk>, Jack Zanker <j.zanker26@imperial.ac.uk>

import numpy as np

def FTBS(phiLevels, params):
    '''
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Solves the linear advection equation for one time step using FTBS.
    phiLevels is updated. Nothing is returned
    Parameters:
        phiLevels ([1darray]): A list length one containing a 1darray which is
                         the latest value of phi, to be updated.
        params (tuple): (c, dt) contains the Courant number and the timestep dt.
    Returns:
        None
    '''
    # Unpack the Courant number from params
    (c, _, _) = params
    # Add another time level to the list, copying from the most recent
    phiLevels.append(phiLevels[-1].copy())
    # Labels for phi at time levels n and n+1
    phiOld = phiLevels[0]
    phiNew = phiLevels[1]
    
    # Update phi for all points
    nx = len(phiOld)
    for i in range(nx):
        phiNew[i] = phiOld[i] - c*(phiOld[i] - phiOld[(i-1)%nx])
    
    # Remove the old value of phiLevels
    phiLevels.pop(0)
    return None

def FTCS(phiLevels, params):
    '''
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Solves the linear advection equation for one time step using FTCS.
    phiLevels is updated. Nothing is returned
    Parameters:
        phiLevels ([1darray]): A list length one containing a 1darray which is
                         the latest value of phi, to be updated.
        params (tuple): (c, dt) contains the Courant number and the timestep dt.
    Returns:
        None
    '''
    # Unpack the Courant number from params
    (c, _, _) = params
    # Add another time level to the list, copying from the most recent
    phiLevels.append(phiLevels[-1].copy())
    # Labels for phi at time levels n and n+1
    phiOld = phiLevels[0]
    phiNew = phiLevels[1]
    
    # Update phi for all points
    nx = len(phiOld)
    for i in range(nx):
        phiNew[i] = phiOld[i] - 0.5*c*(phiOld[(i+1)%nx] - phiOld[(i-1)%nx])
    
    # Remove the old value of phi
    phiLevels.pop(0)
    return None

def CTCS(phiLevels, params):
    '''
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Solves the linear advection equation for one time step using CTCS.
    phiLevels is updated. Nothing is returned.
    If phiLevels contains just one array, then FTCS is used instead.
    Parameters:
        phiLevels ([1darray,1darray]): A list of two consecutive values of phi.
                   If only one is present, the next is created using FTCS.
        params (tuple): (c, dt) contains the Courant number and the timestep dt.
    Returns:
        None
    '''
    # Unpack the Courant number from params
    (c, _, _) = params
    # If phiLevels[1] is not set, calculate it using FTCS
    if len(phiLevels) == 1:
        phiOld = phiLevels[0].copy()
        FTCS(phiLevels, params)
        phiLevels.insert(0, phiOld)
        return None
    
    # Add another time level, copying from the oldest
    phiLevels.append(phiLevels[0].copy())

    # Labels for phi at time levels n-1, n and n+1
    phiOld = phiLevels[0]
    phiMid = phiLevels[1]
    phiNew = phiLevels[2]
    
    # Update phiNew using and phiOld and phiMid
    nx = len(phiOld)
    for i in range(nx):
        phiNew[i] = phiOld[i] - c*(phiMid[(i+1)%nx] - phiMid[(i-1)%nx])
    
    # Remove the old value of phi
    phiLevels.pop(0)
    return None

def Spectral(phiLevels, params):
    '''
    Author: Jack Zanker <j.zanker@imperial.ac.uk>
    Solves the linear advection equation for one time step using the spectral method.
    phiLevels is updated. Nothing is returned
    Parameters:
        phiLevels ([1darray]): A list length one containing a 1darray which is
                         the latest value of phi, to be updated.
        params (tuple): (c, dt) contains the Courant number and the timestep dt.
    Returns:
        None
    '''
    # Unpack the timestep from params
    (_, dt, xLim) = params
    # Add another time level to the list, copying from the most recent
    phiLevels.append(phiLevels[-1].copy())
    # Labels for phi at time levels n and n+1
    phiOld = phiLevels[0]
    phiNew = phiLevels[1]

    # Update phi for all points
    nx = len(phiOld)
    dx = (xLim[1] - xLim[0]) / nx
    k = np.fft.fftfreq(nx, d=dx) * (xLim[1] - xLim[0])
    phi_hat = np.fft.fft(phiOld)
    d_phi_hat = 1j * k * phi_hat
    d_phi = np.fft.ifft(d_phi_hat).real
    phiNew = phiOld - dt * phiOld * d_phi

    # Remove the old value of phi
    phiLevels.pop(0)
    return None












































