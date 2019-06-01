#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is reference numerical Nekrasov expansion for Painlevé V and VI

from pathos.pools import ProcessPool
from functools import partial
from mpmath import mp
mp.dps = 50

try:
  Nexp
except NameError:
  Nexp = 3

try:
  Ntable
except NameError:
  Ntable = 5

def partitions(n):
  a = [0 for i in range(n + 1)]
  k = 1
  y = n - 1
  while k != 0:
    x = a[k - 1] + 1
    k -= 1
    while 2 * x <= y:
      a[k] = x
      y -= x
      k += 1
    l = k + 1
    while x <= y:
      a[k] = x
      a[l] = y
      yield a[:k + 2]
      x += 1
      y -= 1
    a[k] = x + y
    y = x + y - 1
    yield a[:k + 1]


def fricke_jimbo(ps,ss):
  b = mp.fmul(ss[0],ss[1])-mp.fmul(ps[0],ps[2])-mp.fmul(ps[1],ps[3])
  c = ss[0]**2+ss[1]**2+ps[0]**2+ps[1]**2+ps[2]**2+ps[3]**2+ps[0]*ps[1]*ps[2]*ps[3]\
      -(ps[0]*ps[1]+ps[2]*ps[3])*ss[0]-(ps[1]*ps[2]+ps[0]*ps[3])*ss[1]-4.0
  return -b/2.0-mp.sqrt(b**2.0-4.0*c)/2.0

def setarm(partition,pos):
  if pos < len(partition):
    return partition[pos]
  else:
    return 0

def transpose(partition):
  n = len(partition)
  a = [0 for i in range(partition[0])]
  for i in range(partition[0]):
    x = 0
    for j in range(n):
      if partition[j] > i:
        x += 1
    a[i] = x
  return a

def hook_length(partition,i,j):
  if len(partition) > 0:
    trans = transpose(partition)
    return partition[i]+trans[j]-i-j-1
  else:
    return 0

def coefficientV(partition1,partition2,theta,channel):

  b12 = 1.0+0.0j
  for i1 in range(len(partition1)):
    for k1 in range(partition1[i1]):
      b12 *= (theta[2]+channel+2.0*i1-2.0*k1)\
             *((theta[1]+channel+2.0*i1-2.0*k1)**2-theta[0]**2)\
             /8.0/mp.convert(hook_length(partition1,i1,k1))**2\
             /(mp.convert(setarm(transpose(partition1),k1)+setarm(partition2,i1)\
             -i1-k1-1)+channel)**2
  for i2 in range(len(partition2)):
    for k2 in range(partition2[i2]):
      b12 *= (theta[2]-channel+2.0*i2-2.0*k2)\
             *((theta[1]-channel+2.0*i2-2.0*k2)**2-theta[0]**2)\
             /8.0/mp.convert(hook_length(partition2,i2,k2))**2\
             /(mp.convert(setarm(transpose(partition2),k2)+setarm(partition1,i2)\
             -i2-k2-1)-channel)**2
  return b12

def conformal_blockV(theta,channel,level):

    cf12 = 0.0+0.0j
    p = level
    q = 0
    while p >= 0:
      for partition1 in list(partitions(p)):
        for partition2 in list(partitions(q)):
          cf12 += coefficientV(partition1[::-1],partition2[::-1],theta,channel)
      p -= 1
      q += 1
    return cf12

def coefficientVI(partition1,partition2,theta,channel):

  b12 = 1.0+0.0j
  for i1 in range(len(partition1)):
    for k1 in range(partition1[i1]):
      b12 *= ((theta[1]+channel+2.0*i1-2.0*k1)**2-theta[0]**2)\
             *((theta[2]+channel+2.0*i1-2.0*k1)**2-theta[3]**2)\
             /16.0/mp.convert(hook_length(partition1,i1,k1))**2\
             /(mp.convert(setarm(transpose(partition1),k1)+setarm(partition2,i1)\
             -i1-k1-1)+channel)**2
  for i2 in range(len(partition2)):
    for k2 in range(partition2[i2]):
      b12 *= ((theta[1]-channel+2.0*i2-2.0*k2)**2-theta[0]**2)\
             *((theta[2]-channel+2.0*i2-2.0*k2)**2-theta[3]**2)\
             /16.0/mp.convert(hook_length(partition2,i2,k2))**2\
             /(mp.convert(setarm(transpose(partition2),k2)+setarm(partition1,i2)\
             -i2-k2-1)-channel)**2
  return b12

def conformal_blockVI(theta,channel,level):

  cf12 = 0.0+0.0j
  p = level
  q = 0
  while p >= 0:
    for partition1 in list(partitions(p)):
      for partition2 in list(partitions(q)):
        cf12 += coefficientVI(partition1[::-1],partition2[::-1],theta,channel)
    p -= 1
    q += 1
  return cf12

def structure_constantV(theta,channel):

  cn12 = 1.0+0.0j
  for i1 in ( mp.mpc('-1.0'), mp.mpc('1.0') ):
    cn12 *= mp.barnesg(1+theta[2]/2.0+i1*channel/2.0)\
            *mp.barnesg(1+theta[1]/2.0+theta[0]/2.0+i1*channel/2.0)\
            *mp.barnesg(1+theta[1]/2.0-theta[0]/2.0+i1*channel/2.0)\
            /mp.barnesg(1+i1*channel)
  return cn12

def structure_constantVI(theta,channel):

  cn12 = 1.0+0.0j
  for i1 in ( mp.mpc('-1.0'), mp.mpc('1.0') ):
    for j1 in ( mp.mpc('-1.0'), mp.mpc('1.0') ):
      cn12 *= mp.barnesg(1+theta[1]/2.0+j1*theta[0]/2.0+i1*channel/2.0)\
              *mp.barnesg(1+theta[2]/2.0+j1*theta[3]/2.0+i1*channel/2.0)
    cn12 /= mp.barnesg(1+i1*channel)
  return cn12

def expand_tauV_parallel(index,theta,channel,ess):
  return mp.power(ess,index[0]-Nexp)*structure_constantV(theta,channel+2.0*(index[0]-Nexp))*\
         conformal_blockV(theta,channel+2.0*(index[0]-Nexp),index[1])

def expand_tauVI_parallel(index,theta,channel,ess):
  return mp.power(ess,index[0]-Nexp)*structure_constantVI(theta,channel+2.0*(index[0]-Nexp))*\
         conformal_blockVI(theta,channel+2.0*(index[0]-Nexp),index[1])

def essVI(thetas,sigmas,t0=0):
  pis = [ 2.0*mp.cos(mp.pi*thetas[i1]) for i1 in range(len(thetas)) ]
  pijs = [ 2.0*mp.cos(mp.pi*sigmas[i1]) for i1 in range(len(sigmas)) ]

  pijs[2] = fricke_jimbo(pis,pijs)

  if t0 == 0:
    return ((pis[1]*pis[2]+pis[0]*pis[3]-2.0*pijs[1]-pijs[0]*pijs[2])\
           -(pis[0]*pis[2]+pis[1]*pis[3]-2.0*pijs[2]-pijs[0]*pijs[1])\
           *mp.exp(mp.pi*1j*sigmas[0]))\
           /(2*mp.cos(mp.pi*(thetas[1]-sigmas[0]))-pis[0])\
           /(2*mp.cos(mp.pi*(thetas[2]-sigmas[0]))-pis[3])
  else:
    return ((pis[1]*pis[2]+pis[0]*pis[3]-2.0*pijs[1]-pijs[0]*pijs[2])\
           -(pis[0]*pis[2]+pis[1]*pis[3]-2.0*pijs[2]-pijs[0]*pijs[1])\
           *mp.exp(-mp.pi*1j*sigmas[0]))\
           /(2*mp.cos(mp.pi*(thetas[1]-sigmas[0]))-pis[0])\
           /(2*mp.cos(mp.pi*(thetas[2]-sigmas[0]))-pis[3])

def connection_coeff(thetas,sigmas):

  pis = [ 2.0*mp.cos(mp.pi*theta[i1]) for i1 in range(len(theta)) ]
  pijs = [ 2.0*mp.cos(mp.pi*sigma[i1]) for i1 in range(len(sigma)) ]
  pijs[2] = fricke_jimbo(pis,pijs)
  nus[0] = sigmas[0] + thetas[0] + thetas[1]
  nus[1] = sigmas[0] + thetas[2] + thetas[3]
  nus[2] = sigmas[1] + thetas[0] + thetas[3]
  nus[3] = sigmas[1] + thetas[1] + thetas[2]
  mus[0] = thetas[0] + thetas[1] + thetas[2] + thetas[3]
  mus[1] = sigmas[0] + sigmas[1] + thetas[0] + thetas[2]
  mus[2] = sigmas[0] + sigmas[1] + thetas[1] + thetas[3]
  mus[3] = 0.0
  nusigma = (nus[0]+nus[1]+nus[2]+nus[3])/2.0
  q01 = 2.0*pijs[2]+pijs[0]*pijs[1]-pis[0]*pis[2]-pis[1]*pis[3]
  numer = 4.0*mp.sin(mp.pi*sigmas[0])*mp.sin(mp.pi*sigmas[1])\
          +4.0*mp.sin(mp.pi*thetas[1])*mp.sin(mp.pi*thetas[3])\
          +4.0*mp.sin(mp.pi*thetas[0])*mp.sin(mp.pi*thetas[2])
  denom = 0.0
  for i1 in range(4):
    denom += 2.0*(mp.exp(mp.pi*1j*(nusigma-nus[i1]))-mp.exp(mp.pi*1j*(nusigma-mus[i1])))
  omegaplus = 1.0/(2.0*mp.pi*1j)*mp.log((numer+q01)/denom)
  omegaminus = 1.0/(2.0*mp.pi*1j)*mp.log((numer-q01)/denom)
  chi01 = 1.0+0.0j
  for i1 in ( mp.mpc('-1.0'), mp.mpc('1.0') ):
    for j1 in ( mp.mpc('-1.0'), mp.mpc('1.0') ):
       chi01 *= mp.barnesg(1.0+i1*sigmas[1]/2.0+j1*thetas[1]/2.0-i1*j1*thetas[2]/2.0)\
                *mp.barnesg(1.0+i1*sigmas[1]/2.0+j1*thetas[0]/2.0-i1*j1*thetas[3]/2.0)\
                /mp.barnesg(1.0+i1*sigmas[0]/2.0+j1*thetas[1]/2.0+i1*j1*thetas[0]/2.0)\
                /mp.barnesg(1.0+i1*sigmas[0]/2.0+j1*thetas[2]/2.0+i1*j1*thetas[3]/2.0)
    chi01 *= mp.barnesg(1.0+i1*sigmas[0])/mp.barnesg(1.0+i1*sigmas[1])
  for i1 in range(4):
    chi01 *= mp.barnesg(1.0+omegaplus+nus[i1])*mp.barnesg(1.0-omegaplus-mus[i1])\
             /mp.barnesg(1.0-omegaplus-nus[i1])/mp.barnesg(1.0+omegaplus+mus[i1])
  for i1 in ( mp.mpc('-1.0'), mp.mpc('1.0') ):
    for j1 in ( mp.mpc('-1.0'), mp.mpc('1.0') ):
      chi01 *= mp.barnesg(1.0+theta[1]/2.0+j1*theta[0]/2.0+i1*sigmas[0]/2.0)\
               *mp.barnesg(1.0+theta[2]/2.0+j1*theta[3]/2.0+i1*sigmas[0]/2.0)\
               /mp.barnesg(1.0+theta[1]/2.0+j1*theta[2]/2.0+i1*sigmas[1]/2.0)\
               /mp.barnesg(1.0+theta[0]/2.0+j1*theta[3]/2.0+i1*sigmas[1]/2.0)
    chi01 *= mp.barnesg(1.0+i1*sigmas[1])/mp.barnesg(1.0+i1*sigmas[0])
  return chi01

def expand_tauVI(theta,channel,ess):

  indices = [ (a,b) for a in range(2*Nexp+1) for b in range(Ntable) ]
  pool0 = ProcessPool()
  tauVI_parallel = partial( expand_tauVI_parallel, theta=theta, channel=channel, ess=ess)
  taucoefficients = pool0.map( tauVI_parallel, indices )
  return taucoefficients

def expand_tauV(theta,channel,ess):

  indices = [ (a,b) for a in range(2*Nexp+1) for b in range(Ntable) ]
  pool0 = ProcessPool()
  tauV_parallel = partial( expand_tauV_parallel, theta=theta, channel=channel, ess=ess)
  taucoefficients = pool0.map( tauV_parallel, indices )
  return taucoefficients

def schlesinger(n,thetas,sigmas):
  thetas[1] += n
  thetas[3] -= n
  sigmas[0] += n
  sigmas[1] += n
  return thetas, sigmas

def tchannel(th,sig):
  th[0], th[2] = th[2], th[0]
  sig[0], sig[1] = sig[1], sig[0]
  return th,sig

def tauV(coefficients,thetas,channel,t):

  tau11 = 0.0+0.0j

  for i1 in range(2*Nexp+1):
    tau12 = 0.0+0.0j
    for j1 in range(Ntable):
      tau12 += coefficients[i1*Ntable+j1]*mp.power(t,mp.convert(j1))
    tau11 += tau12\
             *mp.power( mp.convert(t),(channel+2.0*mp.convert(i1-Nexp))**2/4.0 )
  return tau11*mp.exp(-mp.convert(t)*thetas[1]/2.0)


def tauVI(coefficients,thetas,channel,t, t0=0):

  if t0 == 0:
    chi = 1.0
  else:
  #  chi = connection_coeff(thetas,sigmas)
    chi = 1.0

  tau11 = 0.0+0.0j

  for i1 in range(2*Nexp+1):
    tau12 = 0.0+0.0j
    for j1 in range(Ntable):
      tau12 += coefficients[i1*Ntable+j1]*mp.power(t,mp.convert(j1))
    tau11 += tau12\
             *mp.power( mp.convert(t),((channel+2.0*mp.convert(i1-Nexp))**2-thetas[0]**2-thetas[1]**2)/4.0 )
  return chi*tau11*mp.power(1.0-mp.convert(t),thetas[1]*thetas[2]/2.0)

def sigmaVI(coefficients,thetas,channel,t):
  def differential(x):
    return mp.log(mp.power( mp.convert(x), (thetas[0]**2+thetas[1]**2-thetas[2]**2-thetas[3]**2)/8.0 )\
           *mp.power( mp.convert(1.0-x), (thetas[1]**2+thetas[2]**2-thetas[0]**2-thetas[3]**2)/8.0 )\
           *tauVI(coefficients,thetas,channel,x))
  return t*(t-1)*mp.diff(differential,t)

def firstlogdiffVI(coefficients,thetas,channel,t):
  def differential(x):
    return mp.log(tauVI(coefficients,thetas,channel,x))
  return mp.diff(differential,t)

def secondlogdiffVI(coefficients,thetas,channel,t):
  def differential(t):
    return t*(t-1.0)*firstlogdiffVI(coefficients,thetas,channel,t)
  return mp.diff(differential,t)

def accessoryKVI(coefficients,thetas,channel,t):
  return firstlogdiffVI(coefficients,thetas,channel,t)-thetas[1]*(thetas[0]/t+thetas[2]/(t-1))/2.0

def todaFVI(coefficients,thetas,sigmas,t):
  return secondlogdiffVI(coefficients,thetas,channel,t)-(thetas[3]-thetas[1])*thetas[1]/2.0

def firstlogdiffV(coefficients,thetas,channel,t):
  def differential(x):
    return mp.log(tauV(coefficients,thetas,channel,x))
  return mp.diff(differential,t)

def secondlogdiffV(coefficients,thetas,channel,t):
  def differential(x):
    return t*firstlogdiffV(coefficients,thetas,channel,x)
  return mp.diff(differential,t)

def accessoryKV(coefficients,thetas,channel,t):
  return firstlogdiffV(coefficients,thetas,channel,t)+theta[0]*theta[1]

def todaFV(coefficients,thetas,channel,t):
  return secondlogdiffV(coefficients,thetas,channel,t)+theta[1]/2.0
