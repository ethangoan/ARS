"""
Created on Fri Mar 27 06:46:02 2015

@author: John Grenall (original author), Alberto Lumbreras (reviewed and cleaned)
         updated by Ethan Goan to work with Python3 and Tensorflow

"""

import numpy as np
import random
from matplotlib import pyplot as plt

import tensorflow as tf

class ARS():
  '''
  This class implements the Adaptive Rejection Sampling technique of Gilks and Wild '92.
  Where possible, naming convention has been borrowed from this paper.
  The PDF must be log-concave.

  Currently does not exploit lower hull described in paper- which is fine for drawing
  only small amount of samples at a time.
  '''

  def __init__(self, f, fprima, xi=[-4,1,4], lb=-np.Inf, ub=np.Inf, use_lower=False, ns=50, **fargs):
    '''
    initialize the upper (and if needed lower) hulls with the specified params

    Parameters
    ==========
    f: function that computes log(f(u,...)), for given u, where f(u) is proportional to the
       density we want to sample from
    fprima:  d/du log(f(u,...))
    xi: ordered vector of starting points in wich log(f(u,...) is defined
      to initialize the hulls
    D: domain limits
    use_lower: True means the lower sqeezing will be used; which is more efficient
           for drawing large numbers of samples


    lb: lower bound of the domain
    ub: upper bound of the domain
    ns: maximum number of points defining the hulls
    fargs: arguments for f and fprima
    '''

    self.lb = tf.Variable(lb, dtype=tf.float32)
    self.ub = tf.Variable(ub, dtype=tf.float32)
    self.f = f
    self.fprima = fprima
    self.fargs = fargs

    #set limit on how many points to maintain on hull
    self.ns = 100
    self.x_t = tf.Variable(xi[1], dtype=tf.float32) # initialize x, the vector of absicassae at which the function h has been evaluated
    self.x = np.array(xi)
    self.h = np.array([])
    self.hprime = np.array([])
    self.update_log_energy()
    self.h_lb = self.f(self.lb, **self.fargs)
    self.h_ub = self.f(self.ub, **self.fargs)
    self.hprime_lb = self.fprima(self.lb, **self.fargs)
    self.hprime_ub = self.fprima(self.ub, **self.fargs)

    #Avoid under/overflow errors. the envelope and pdf are only
    # proporitional to the true pdf, so can choose any constant of proportionality.
    self.offset = np.amax(self.h)
    print('self.offset = {}'.format(self.offset))
    self.h = self.h-self.offset

    # Derivative at first point in xi must be > 0
    # Derivative at last point in xi must be < 0
    print('hprime_lb = {}'.format(self.hprime_lb))
    print('hprime_ub = {}'.format(self.hprime_ub))

    if not(self.hprime_lb > 0):
      print(self.hprime)
      raise IOError('initial anchor points must span mode of PDF')
    if not(self.hprime_ub < 0):
      print(self.hprime)
      raise IOError('initial anchor points must span mode of PDF')
    self.insert()


  def update_log_energy(self):
    """append new log energy and grad calls"""
    # need to make sure that the number of elements in self.x
    # are greater than what we currently have. Otherwise we
    # won't need to update anything, and if we have called it
    # it likely means there is an error in our logic.
    if(len(self.h) > len(self.x)) or (len(self.hprime) > len(self.x)):
      raise RuntimeError('Length of log energy or grad is not less than x')
    # now evaluate the log energyu and the log gradient
    # for the new values of x
    new_h = []
    new_hprime = []
    for x_idx in range(self.h.size, self.x.size):
      new_h.append(self.f(self.x[x_idx], **self.fargs))
      new_hprime.append(self.fprima(self.x[x_idx], **self.fargs))
    # add the new log energy and gradients to the class attributes
    self.h = np.hstack([self.h, np.array(new_h).ravel()])
    self.hprime = np.hstack([self.hprime, np.array(new_hprime).ravel()])


  def draw(self, N):
    '''
    Draw N samples and update upper and lower hulls accordingly
    '''
    samples = np.zeros(N)
    n=0
    while n < N:
      [xt,i] = self.sampleUpper()
      # TODO (John): Should perform squeezing test here but not yet implemented
      ht = self.f(xt, **self.fargs)
      hprimet = self.fprima(xt, **self.fargs)
      ht = ht - self.offset
      #ut = np.amin(self.hprime*(xt-x) + self.h);
      ut = self.h[i] + (xt-self.x[i])*self.hprime[i]

      # Accept sample? - Currently don't use lower
      u = random.random()
      if u < np.exp(ht-ut):
        samples[n] = xt
        n +=1

      # Update hull with new function evaluations
      if self.u.__len__() < self.ns:
        self.insert([xt],[ht],[hprimet])

    return samples


  def insert(self,xnew=[],hnew=[],hprimenew=[]):
    '''
    Update hulls with new point(s).
    If none given, just recalculate hull from existing x,h,hprime
    '''
    if xnew.__len__() > 0:
      x = np.hstack([self.x,xnew])
      # getting the an index of our input variables to make
      # sure they are in ascending order
      idx = np.argsort(x)
      # apply idx to make sure in correct order
      self.x = x[idx]
      self.h = np.hstack([self.h, np.array(hnew).ravel()])[idx]
      self.hprime = np.hstack([self.hprime, np.array(hprimenew).ravel()])[idx]

    # we need to determine where piecewise-function components intersect
    # for models evaluated at abscissae
    self.z = np.zeros(len(self.x)+1)
    # This is the formula explicitly stated in Gilks.
    # Requires 7(N-1) computations
    # Following line requires 6(N-1)
    # self.z[1:-1] = (np.diff(self.h) + self.x[:-1]*self.hprime[:-1] - self.x[1:]*self.hprime[1:]) / -np.diff(self.hprime);
    print(np.array(self.x))
    print(np.array(self.x).shape)
    print(np.array(self.h).shape)
    h = np.array(self.h).ravel()
    x = np.array(self.x).ravel()
    hprime = np.array(self.hprime).ravel()

    self.z[1:-1] = (np.diff(h) - np.diff(np.array(x)*np.array(hprime)))/-np.diff(hprime)
    self.z[0] = self.lb.numpy(); self.z[-1] = self.ub.numpy()
    N = self.h.__len__()
    self.u = np.zeros(len(self.x))
    l_range = list(range(N))
    self.u = hprime[[0]+l_range]*(self.z-x[[0]+l_range]) + h[[0]+l_range]
    self.s = np.hstack([0,np.cumsum(np.diff(np.exp(self.u))/hprime)])
    print(self.s)
    self.cu = self.s[-1]



  def sampleUpper(self):
    '''
    Return a single value randomly sampled from the upper hull and index of segment
    '''
    u = random.random()

    # Find the largest z such that sc(z) < u
    print('here')
    print((self.s/self.cu))
    i = np.nonzero(self.s/self.cu < u)[0][-1]
    # Figure out x from inverse cdf in relevant sector
    xt = self.x[i] + (-self.h[i] + np.log(self.hprime[i]*(self.cu*u - self.s[i]) +
    np.exp(self.u[i]))) / self.hprime[i]

    return [xt,i]


  def plotHull(self):
    '''
    Plot the piecewise linear hull using matplotlib
    '''

    xpoints = self.z
    #ypoints = np.hstack([0,np.diff(self.z)*self.hprime])
    ypoints = np.exp(self.u)
    plt.plot(xpoints,ypoints)
    plt.show()
    '''
    for i in range(1,self.z.__len__()):
      x1 = self.z[i]
      y1 = 0
      x2 = self.z[i+1]
      y2 = self.z[i+1]-self.z[i] * hprime[i]
    '''
