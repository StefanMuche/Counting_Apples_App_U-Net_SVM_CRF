# -*- coding: utf-8 -*-
"""
belief_propagation.py

Created on Fri Mar 31 16:05:36 2017

@author: DamianCristian
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import rescale, resize

if __name__ == '__main__':
    from skimage.data import horse
    from skimage.morphology import dilation
    from time import process_time as clock
    from time import sleep
    from matplotlib import pyplot as plt
    #from graph_cuts_regularizer import regularize_abs_dif

def gen_data_belief(data, labels, cost_fun):
    if not data.ndim == 2:
        raise ValueError("`data` needs to be a 2-dimensional array") 
               
    belief = np.empty(data.shape + (len(labels),))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(len(labels)):
                belief[i,j,k]= cost_fun(labels[k],data[i,j])
                
    return belief

neighbors = [(-1,0),(0,-1),(0,1),(1,0)]

             
def iter_neigh(b,i,j):
    for nx, ny in neighbors:
        if 0 <= i+nx < b.shape[0]:
            if 0 <= j+ny < b.shape[1]:
                yield (i+nx,j+ny)
        
def choose(index , choices, out=None):
    if out is None:
        out = np.empty(index.shape, choices.dtype)
    for it in np.ndindex(index.shape):
        out[it] = choices[it+(index[it],)]
    return out 
    
gradk = np.array([0,1,-1])

    
def potts_energy(l, D, smooth_cost):
    smooth_cost = np.squeeze(smooth_cost)
    costl = np.sum( smooth_cost*(ndi.correlate1d(l, gradk, 1) != 0))
    costr = np.sum( smooth_cost*(ndi.correlate1d(l, np.flipud(gradk), 1) != 0))
    costd = np.sum( smooth_cost*(ndi.correlate1d(l, gradk, 0) != 0))    
    costu = np.sum( smooth_cost*(ndi.correlate1d(l, np.flipud(gradk), 0) != 0))
    dci = choose(l, D)
    return np.sum(dci) + costl + costr + costu + costd
    

def solve_potts(data_cost, smooth_cost, niter=50):
    """
    Finds the MAP estimate of Markov random field with a square grid topology 
    using the Potts model for the smoothness term.
    """
    
    ndt = data_cost - np.min(data_cost, 2, keepdims=True)
    
    b = np.copy(ndt)
    received = np.empty_like(b)
    
    
    
    for _ in range(niter):    
        # Exchange messages
        for  i, j, k in np.ndindex(b.shape):
            received[i,j,k] = 0
            for x, y in iter_neigh(b,i,j):
                message = min(b[x, y, k], smooth_cost(i, j, x, y))
                received[i, j, k] += message
        
        np.copyto(b, ndt)
        b += received
       
        # Normalisation (min of b is always 0)
        b -= np.min(b,2,keepdims=True)
       
                
            
    return np.argmin(b, 2)                
                
                
def solve_homogeneous_potts(data_cost, smooth_cost, niter=50 , init_belief=None, get_belief=False):
    """
    Finds the MAP estimate of Markov random field with a square grid topology 
    using the Potts model for the smoothness term.
    """
    
    if np.ndim(smooth_cost) == 2:
        smooth_cost = smooth_cost.reshape(smooth_cost.shape + (1,))
            
    ndt = data_cost - np.min(data_cost, 2, keepdims=True)
    
    if init_belief is None:
        b = np.copy(ndt)
    else:
        b = init_belief
        
    struct = np.array([[[0],[1],[0]],[[1],[0],[1]],[[0],[1],[0]]])
    received = np.empty_like(b)
    message = np.empty_like(b)
    
    for i in range(niter):
        # Calculate messages min(b, min(b,2)+smooth_cost)
        np.minimum(b, smooth_cost, out=message)
        # Exchange messages 
        ndi.convolve(message, struct, output= received)
        # Update
        #np.copyto(b, ndt)
        b += received
        # Normalisation (min(b,20 is always 0)
        b -= np.min(b,2,keepdims=True)
        #b /= 5
        

    if get_belief is True:
        return b
                        
    return np.argmin(b,2)
    
    
def solve_potts_pyramid(dtb, smooth, factor):   
    
    idtb = rescale(dtb, 2**(-factor))
    ib = solve_homogeneous_potts(idtb, smooth, 100, get_belief=True)
    
    for i in range(factor):
        ib = solve_homogeneous_potts(idtb, smooth, 100, init_belief=ib, get_belief=True)
        ib = rescale(ib, 2)
        idtb = rescale(idtb, 2)
    
    ib = resize(ib, dtb.shape) 
    
    return solve_homogeneous_potts(dtb, smooth, 1000, init_belief=ib)
        
    
def _demo_potts_():
    
    oi = np.array(horse(),dtype=bool)
    lb = [False, True]
    
    ni = np.random.binomial(1,.02,oi.shape)
    ni = dilation(ni)
    dt = np.logical_xor(oi,ni)
 
    data_cost_f = lambda x1, x2: 0.5 * (not x1 == x2)
    dtb = gen_data_belief(dt, lb, data_cost_f)
    print(dtb)
    
    plt.figure('Data')
    plt.imshow(dt, interpolation='none')

    plt.show()
    
    print('Method 1')

    #    starttime = clock()
    #    et1 = regularize_abs_dif(dtb, 1)
    #    runtime = clock() - starttime

    starttime = clock()
    et1 = solve_potts(dtb, lambda a,b,c,d:10, 10)
    runtime = clock() - starttime

    print('Runtime:', runtime)
    print('Error:', np.mean(abs(et1-oi)))

    plt.figure('Estimate method 1')
    plt.imshow(et1, interpolation='none')
    plt.show()
       
    print('Method 2')
    
    starttime = clock()
    et2 = solve_homogeneous_potts(dtb, 1, niter=1000)
    runtime = clock() - starttime

    print('Runtime:', runtime)
    print('Error:', np.mean(abs(et2-oi)))
    
    plt.figure('Estimate method 2')
    plt.imshow(et2, interpolation='none')
    #plt.show()

    
    plt.figure('Original')
    plt.imshow(oi, interpolation='none')
    plt.show()
    
if __name__ == '__main__':
    _demo_potts_()
    
    
