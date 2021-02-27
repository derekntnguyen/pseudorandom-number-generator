#  -*- coding: utf-8 -*-
'''
P1 & P2 Assignment
Derek Nguyen
Created: 2019-01-29
Modified: 2019-01-31
Due: 2019-01-31
'''

# %% codecell
#P1a

import numpy as np

from mpl_toolkits import mplot3d
import matplotlib# used to create interactive plots in the Hydrogen package of the Atom IDE
matplotlib.use('Qt5Agg') # used to create interactive plots in the Hydrogen package of the Atom IDE
import matplotlib.pyplot as plt

import os
import random

def p1(size = None, method = 'NR', seed = None, returnSeed = False):
    '''
    Pseudorandom numbers generated between [0 1]
    Uses a linear congruential generator algorithm to generate a vector or array of pseudo random numbers from [0 1]
    Parameter values for the generator depend on the specified:
    size, an integer, tuple, or NONE
    method, a string that is either 'NR' (Numerical Recipes) or 'RANDU' (the RANDU generator)
    seed, an integer or NONE
    returnSeed, as a boolean True or False, or NONE
    Derek Nguyen
    Created: 01/31/2020
    Last revised: 01/31/2020
    '''

    if size is None: #determines the length of the array created depending on the size passed to the function, if none, 1 random number is returned
        n = 1
    elif isinstance(size, int):
        n = size
    elif isinstance(size, tuple):
        n = np.prod(size)
    else:
        raise Exception('Error - size is not an integer or tuple')


    if method == 'NR': #specifies the method for which to calculate the psuedo random numbers, either NR or RANDU
        a = 1664525
        b = 1013904223
        c = 2**32
    elif method == 'RANDU':
        a = 65539
        b = 0
        c = 2**31
    else:
        raise Exception('Error - Select either NR or RANDU')


    if seed is None: #determines if a seed is passed to the function
        seeder = int.from_bytes(os.urandom(20), 'big') # if no seed is passed, a cryptographic level seed is created using os.urandom
        seed = seeder % c
    elif isinstance(seed, int):
        pass
    else:
        raise Exception('Error - not a valid integer')


    y = np.zeros(n) #preallocates the array
    y[0] = seed


    for i in range(1,n): #utilizing the specified method, creates an array with y[0] being the seed
        y[i] = (a * y[i-1] + b) % c

    y = y/c
    y = np.reshape(y, (size)) #reshapes the array to the original specified size passed to the function


    if returnSeed is True: #determines whether or not to return the seed used in the algorithm
        return seed, y
    elif returnSeed is False:
        return y
    else:
        raise Exception('Error - not a valid returnSeed input')

#P1b
nrarray = p1((5000,3), method = 'NR') #creates an array utilizing the nr method and assigns each column to x, y, and z
randuarray = p1((5000,3), method = 'RANDU') #creates an array utilizing the randu method and assigns each column to x, y, and z
xnumpy = np.random.random_sample(5000) ##creates an array utilizing the numpy random method and assigns it to x, y, and z
ynumpy = np.random.random_sample(5000)
znumpy = np.random.random_sample(5000)


plt.figure(1) #creates a 3d scatter plot of the nr method
xnr = nrarray[:,0]
ynr = nrarray[:,1]
znr = nrarray[:,2]
axnr = plt.axes(projection='3d')
axnr.scatter3D(xnr,ynr,znr);
plt.title('NR Method')


plt.figure(2) #creates a 3d scatter plot of the randu method
xrandu = randuarray[:,0]
yrandu = randuarray[:,1]
zrandu = randuarray[:,2]
axrandu = plt.axes(projection='3d')
axrandu.scatter3D(xrandu,yrandu,zrandu);
plt.title('RANDU Method')


plt.figure(3) #creates a 3d scattor plot of the numpy random method
axnumpy = plt.axes(projection='3d')
axnumpy.scatter3D(xnumpy,ynumpy,znumpy);
plt.title('Numpy Method')

#P1d
rows = 5000
cols = 3
seedtest = np.zeros((rows,cols)) #preallocates an array of zeros
for row in range(rows): #utilizes a for loop to iterate and place a seed values in the array
    for col in range(cols):
        seedtest[row,col] = int.from_bytes(os.urandom(20), 'big')

plt.figure(4) #creates a 3d scattor plot of seed values
xseed = seedtest[:,0]
yseed = seedtest[:,1]
zseed = seedtest[:,2]
axseed = plt.axes(projection='3d')
axseed.scatter3D(xseed,yseed,zseed);
plt.title('Seed Test')


# %% codecell
#P2
def p2(nThrows, method = 'NR'):

    if method == 'NR': #specifies the method for which to calculate the psuedo random numbers, either NR or RANDU
        a = 1664525
        b = 1013904223
        c = 2**32
    elif method == 'RANDU':
        a = 65539
        b = 0
        c = 2**31
    else:
        raise Exception('Error - Select either NR or RANDU')

    seeder = int.from_bytes(os.urandom(20), 'big') # if no seed is passed, a cryptographic level seed is created using os.urandom
    seed = seeder % c

    x = np.zeros(nThrows)
    y = np.zeros(nThrows) #preallocates the array
    x[0] = seed
    y[0] = seed

    for i in range(1,nThrows): #utilizing the specified method, creates an array with y[0] being the seed
        x[i] = (a * x[i-1] + b) % c
        y[i] = (a * y[i-1] + b) % c

    x = x/c
    y = y/c
    circle = np.sqrt(x**2 + y**2)
    nInside = 0

    for throws in range(len(circle)):
        if circle[throws] <= 1:
            nInside += 1

    piest = 4 * nInside / nThrows
    return piest

print(p2(5000, method = 'NR'))

nThrows = 5000
a = 1664525
b = 1013904223
c = 2**32
seeder = int.from_bytes(os.urandom(20), 'big') # if no seed is passed, a cryptographic level seed is created using os.urandom
seed = seeder % c
x = np.zeros(nThrows)
y = np.zeros(nThrows) #preallocates the array
x[0] = seed
y[0] = seed
for i in range(1,nThrows): #utilizing the specified method, creates an array with y[0] being the seed
    x[i] = (a * x[i-1] + b) % c
    y[i] = (a * y[i-1] + b) % c
x = np.random.random(nThrows)
y = np.random.random(nThrows)
circle = np.sqrt(x**2 + y**2)
nInside = 0

for throws in range(len(circle)):
    if circle[throws] <= 1:
        nInside += 1

piest = 4 * nInside / nThrows
print(piest)
