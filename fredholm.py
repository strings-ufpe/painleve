#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathos.pools import ProcessPool
from functools import partial
from mpmath import mp
import numpy as np
mp.dps = 50
h_diff = 1e-20

try:
  N
except NameError:
  N = 32

def fricke_jimbo(ps,ss):
  b = mp.fmul(ss[0],ss[1])-mp.fmul(ps[0],ps[2])-mp.fmul(ps[1],ps[3])
  c = ss[0]**2+ss[1]**2+ps[0]**2+ps[1]**2+ps[2]**2+ps[3]**2+ps[0]*ps[1]*ps[2]*ps[3]\
      -(ps[0]*ps[1]+ps[2]*ps[3])*ss[0]-(ps[1]*ps[2]+ps[0]*ps[3])*ss[1]-4.0
  return -b/2.0-mp.sqrt(b**2.0-4.0*c)/2.0

def computeeta(thetas,sigmas,t0=0):
  pis = [ 2.0*mp.cos(mp.pi*thetas[i1]) for i1 in range(len(thetas)) ]
  pijs = [ 2.0*mp.cos(mp.pi*sigmas[i1]) for i1 in range(len(sigmas)) ]

  pijs[2] = fricke_jimbo(pis,pijs)

  # this computes eeta even when ess is not well defined

  if t0 == 0:
    prefactor = (pis[1]*pis[2]+pis[0]*pis[3]-2.0*pijs[1]-pijs[0]*pijs[2])\
           -(pis[0]*pis[2]+pis[1]*pis[3]-2.0*pijs[2]-pijs[0]*pijs[1])\
           *mp.exp(mp.pi*1j*sigmas[0])
  else:
    prefactor = (pis[1]*pis[2]+pis[0]*pis[3]-2.0*pijs[1]-pijs[0]*pijs[2])\
           -(pis[0]*pis[2]+pis[1]*pis[3]-2.0*pijs[2]-pijs[0]*pijs[1])\
           *mp.exp(-mp.pi*1j*sigmas[0])
  prefactor *= mp.gamma((sigmas[0]-thetas[0]-thetas[1])/2)\
               *mp.gamma((sigmas[0]-thetas[2]-thetas[3])/2)
  prefactor /= 16*mp.pi**2*mp.sin(mp.pi*(sigmas[0]+thetas[0]-thetas[1])/2)\
               *mp.sin(mp.pi*(sigmas[0]+thetas[3]-thetas[2])/2)
  return prefactor*(mp.gamma(1-sigmas[0])**2/mp.gamma(1+sigmas[0])**2)\
         *mp.gamma(1+(thetas[1]+thetas[0]+sigmas[0])/2)*mp.gamma(1+(thetas[1]-thetas[0]+sigmas[0])/2)\
         *mp.gamma(1+(thetas[2]+thetas[3]+sigmas[0])/2)*mp.gamma(1+(thetas[2]-thetas[3]+sigmas[0])/2)\
         /(mp.gamma(1+(thetas[1]-thetas[0]-sigmas[0])/2)*mp.gamma(1+(thetas[2]-thetas[3]-sigmas[0])/2))

def esshatV(thetas,sigma,ess):
  return ess*(mp.gamma(1-sigma)**2/mp.gamma(1+sigma))**2\
            *mp.gamma(1+(sigma+thetas[2])/2)*mp.gamma(1+(sigma+thetas[1]+thetas[0])/2)*mp.gamma(1+(sigma+thetas[1]-thetas[0])/2)\
            *mp.gamma(1-(sigma-thetas[2])/2)*mp.gamma(1-(sigma-thetas[1]-thetas[0])/2)*mp.gamma(1-(sigma-thetas[1]+thetas[0])/2)

def tchannel(th,sig):
  th[0], th[2] = th[2], th[0]
  sig[0], sig[1] = sig[1], sig[0]
  return th,sig

def determinant(A):
  def reduce(A):
    index = [ p for p in range(len(A)) ]
    pool = ProcessPool()
    B = np.array(map( lambda index: A[index] - A[index,0]/A[0,0]*A[0], index ))
    return B[1:,1:]
  if len(A) == 1:
    return A[0,0]
  else:
    p = 0
    fac = 1
    while np.absolute(A[p,0]) < 1.0e-17:
      p += 1
    if p == len(A):
      return 0
    if p > 0:
      A[0], A[p] = A[p], A[0]
      p += 1
      fac = -fac
    else:
      return fac*A[0,0]*determinant(reduce(A))

def invertseries(seq):
  result = np.zeros(shape=(N+1,2,2),dtype=object)
  result[0] = [ [ mp.convert(1.0) , mp.convert(0.0) ], [mp.convert(0.0), mp.convert(1.0) ] ]
  for p in range(1,N+1):
    A = -mp.matrix(seq[p])
    for q in range(1,p):
      A += -mp.matrix(seq[q])*mp.matrix(result[p-q])
    result[p] = A.tolist()
  return result

def gee(th1,th2,th3):
  psi = np.empty(shape=(N+1,2,2),dtype=object)
  a = (th1-th2+th3)/2
  b = (th1-th2-th3)/2
  c = th1
  # psi is actually invariant by conjugation by a diagonal matrix
  psi[0] = [ [ mp.convert(1.0) , mp.convert(0,0) ], [mp.convert(0,0), mp.convert(1.0) ] ]
  psi[1] = [ [ a*b/c, -a*b/c/(1+c) ], [ (c-a)*(c-b)/c/(1-c), -(a-c)*(b-c)/c ] ]
  for p in range(2,N+1):
    psi[p,0,0] = (a+p-1)*(b+p-1)/((c+p-1)*p)*psi[p-1,0,0]
    psi[p,0,1] = (a+p-1)*(b+p-1)/((c+p)*(p-1))*psi[p-1,0,1]
    psi[p,1,0] = (a-c+p-1)*(b-c+p-1)/((-c+p)*(p-1))*psi[p-1,1,0]
    psi[p,1,1] = (a-c+p-1)*(b-c+p-1)/((-c+p-1)*p)*psi[p-1,1,1]
  return psi

def geeV(th1,th2):
  psi = np.empty(shape=(N+1,2,2),dtype=object)
  a = (th1+th2)/2.0
  c = th1
  psi[0] = [ [ mp.convert(1.0) , mp.convert(0,0) ], [mp.convert(0,0), mp.convert(1.0) ] ]
  psi[1] = [ [ a/c, -(a-c)/(c*(c+1)) ], [ a/(c*(1-c)) , -(a-c)/c ]]
  for p in range(2,N+1):
    psi[p,0,0] = (a+p-1)/((c+p-1)*p)*psi[p-1,0,0]
    psi[p,0,1] = (a+p-1)/((c+p)*(p-1))*psi[p-1,0,1]
    psi[p,1,0] = (a-c+p-1)/((-c+p)*(p-1))*psi[p-1,1,0]
    psi[p,1,1] = (a-c+p-1)/((-c+p-1)*p)*psi[p-1,1,1]
  return psi

def buildA_parallel(index,vecg,vecginv):
  result = mp.matrix(2)
  if index[1] < index[0]+1:
    for r in range(N-index[0]):
      result += -mp.matrix(vecg[N-1-index[0]-r])*mp.matrix(vecginv[index[1]+r+1])
  return result.tolist()

def buildA(thetas,sigma):
  vecg = gee(sigma,thetas[2],thetas[3])
  vecginv = invertseries(vecg)
  index = [ (p,q) for p in range(N) for q in range(N) ]
  Aparallel = partial( buildA_parallel, vecg=vecg, vecginv=vecginv )
  pool = ProcessPool()
  OPA = np.array(pool.map( Aparallel, index ))
  bigA = np.empty(shape=(2*N,2*N),dtype=object)
  for p in range(N):
    for q in range(N):
      bigA[2*p:2*p+2,2*q:2*q+2] = OPA[N*p+q]
  return bigA

def buildAV(thetas,sigma):
  vecg = geeV(sigma,thetas[2])
  vecginv = invertseries(vecg)
  index = [ (p,q) for p in range(N) for q in range(N) ]
  Aparallel = partial( buildA_parallel, vecg=vecg, vecginv=vecginv )
  pool = ProcessPool()
  OPA = np.array(pool.map( Aparallel, index ))
  bigA = np.empty(shape=(2*N,2*N),dtype=object)
  for p in range(N):
    for q in range(N):
      bigA[2*p:2*p+2,2*q:2*q+2] = OPA[N*p+q]
  return bigA

def buildD_parallel(index,vecg,vecginv):
  result = mp.matrix(2)
  if index[1] > index[0]-1:
    for r in range(index[0]+1):
      result += -mp.matrix(vecg[index[0]-r])*mp.matrix(vecginv[N-index[1]+r])
  return result.tolist()

def buildD(thetas,sigma):
  vecg = gee(-sigma,thetas[1],thetas[0])
  vecginv = invertseries(vecg)
  index = [ (p,q) for p in range(N) for q in range(N) ]
  Dparallel = partial( buildD_parallel, vecg=vecg, vecginv=vecginv)
  pool = ProcessPool()
  OPD = np.array(pool.map( Dparallel, index ))
  bigD = np.empty(shape=(2*N,2*N),dtype=object)
  for p in range(N):
    for q in range(N):
      bigD[2*p:2*p+2,2*q:2*q+2] = OPD[N*p+q]
  return bigD

def multiplyD(OPD,sigma,eeta,t):
  tmatrix = mp.matrix(2)
  tmatrix[0,0] = mp.power(t,-sigma/2.0)/mp.sqrt(eeta)
  tmatrix[1,1] = mp.power(t,sigma/2.0)*mp.sqrt(eeta)
  tmatrixinv = mp.matrix(2)
  tmatrixinv[0,0] = mp.power(t,sigma/2.0)*mp.sqrt(eeta)
  tmatrixinv[1,1] = mp.power(t,-sigma/2.0)/mp.sqrt(eeta)
  result = np.empty(shape=(2*N,2*N),dtype=object)
  for p in range(N):
    for q in range(N):
      A = mp.power(t,N+p-q)*tmatrix*mp.matrix(OPD[2*p:2*p+2,2*q:2*q+2])*tmatrixinv
      result[2*p:2*p+2,2*q:2*q+2] = A.tolist()
  return result

def multiplyDdiff(OPD,sigma,eeta,t):
  tmatrix = mp.matrix(2)
  tmatrix[0,0] = mp.power(t,-sigma/2.0)/mp.sqrt(eeta)
  tmatrix[1,1] = mp.power(t,sigma/2.0)*mp.sqrt(eeta)
  tmatrixinv = mp.matrix(2)
  tmatrixinv[0,0] = mp.power(t,sigma/2.0)*mp.sqrt(eeta)
  tmatrixinv[1,1] = mp.power(t,-sigma/2.0)/mp.sqrt(eeta)
  sigma3 = mp.matrix(2)
  sigma3[0,0] = 1.0
  sigma3[1,1] = -1.0
  result = np.empty(shape=(2*N,2*N),dtype=object)
  for p in range(N):
    for q in range(N):
      X = mp.power(t,N+p-q)*tmatrix*mp.matrix(OPD[2*p:2*p+2,2*q:2*q+2])*tmatrixinv
      A = (N+p-q)*X/t-sigma/(2.0*t)*(sigma3*X-X*sigma3)
      result[2*p:2*p+2,2*q:2*q+2] = A.tolist()
  return result

def multiplyDdiff2(OPD,sigma,eeta,t):
  tmatrix = mp.matrix(2)
  tmatrix[0,0] = mp.power(t,-sigma/2.0)/mp.sqrt(eeta)
  tmatrix[1,1] = mp.power(t,sigma/2.0)*mp.sqrt(eeta)
  tmatrixinv = mp.matrix(2)
  tmatrixinv[0,0] = mp.power(t,sigma/2.0)*mp.sqrt(eeta)
  tmatrixinv[1,1] = mp.power(t,-sigma/2.0)/mp.sqrt(eeta)
  sigma3 = mp.matrix(2)
  sigma3[0,0] = 1.0
  sigma3[1,1] = -1.0
  result = np.empty(shape=(2*N,2*N),dtype=object)
  for p in range(N):
    for q in range(N):
      X = mp.power(t,N+p-q)*tmatrix*mp.matrix(OPD[2*p:2*p+2,2*q:2*q+2])*tmatrixinv
      A = (N+p-q)*(N+p-q-1.0)*X/t**2+sigma**2/(2.0*t**2)*(X-sigma3*X*sigma3)\
          +(sigma/(2.0*t**2)-(N+p-q)*sigma/(t**2))*(sigma3*X-X*sigma3)
      result[2*p:2*p+2,2*q:2*q+2] = A.tolist()
  return result

def tauV(OPA,OPD,thetas,sigma,esshat,t):
  OPK = mp.matrix(OPA)*mp.matrix(multiplyD(OPD,sigma,esshat,t))
  value = mp.power(t,(sigma**2-thetas[2]**2)/4.0)*mp.exp(-thetas[1]*t)\
          *mp.det(mp.eye(2*N)-OPK)
  return value

def tauVI(OPA,OPD,thetas,sigma,eeta,t):
  OPK = mp.matrix(OPA)*mp.matrix(multiplyD(OPD,sigma,eeta,t))
  value = mp.power(t,(sigma**2-thetas[0]**2-thetas[1]**2)/4.0)\
        *mp.power(1-t,-thetas[1]*thetas[2]/2.0)\
        *mp.det(mp.eye(2*N)-OPK)
  return value

def sigmaVI(OPA,OPD,thetas,channel,eeta,t):
  def differential(x):
    return mp.log(mp.power( mp.convert(x), (thetas[0]**2+thetas[1]**2-thetas[2]**2-thetas[3]**2)/8.0 )\
           *mp.power( mp.convert(1.0-x), (thetas[1]**2+thetas[2]**2-thetas[0]**2-thetas[3]**2)/8.0 )\
           *tauVI(OPA,OPD,thetas,channel,eeta,x))
  return t*(t-1)*mp.diff(differential,t)

def schlesinger(n,thetas,sigmas):
  thetas[1] += n
  thetas[3] -= n
  sigmas[0] += n
  sigmas[1] += n
  return thetas, sigmas

def firstlogdiffVInum(OPA,OPD,thetas,channel,eeta,t):
  def differential(x):
    return mp.log(tauVI(OPA,OPD,thetas,channel,eeta,x))
  return mp.diff(differential,t,relative = True)

def firstlogdiffVI(OPA,OPD,thetas,channel,eeta,t):
  OPK = mp.matrix(OPA)*mp.matrix(multiplyD(OPD,channel,eeta,t))
  OPKdiff = mp.matrix(OPA)*mp.matrix(multiplyDdiff(OPD,channel,eeta,t))
  BIGM = -(mp.eye(2*N)-OPK)**-1*mp.matrix(OPKdiff)
  value = 0.0
  for i in range(2*N) :
    value += BIGM[i,i]
  return (channel**2-thetas[0]**2-thetas[1]**2)/(4.0*t)\
          -thetas[1]*thetas[2]/(2.0*(t-1.0))+value

def secondlogdiffVInum(OPA,OPD,thetas,channel,eeta,t):
  def differential(t):
    return t*(t-1.0)*firstlogdiffVInum(OPA,OPD,thetas,channel,eeta,t)
  return mp.diff(differential,t,relative = True)

def secondlogdiffVI(OPA,OPD,thetas,channel,eeta,t):
  OPK = mp.matrix(OPA)*mp.matrix(multiplyD(OPD,channel,eeta,t))
  OPKdiff = mp.matrix(OPA)*mp.matrix(multiplyDdiff(OPD,channel,eeta,t))
  OPKdiff2 = mp.matrix(OPA)*mp.matrix(multiplyDdiff2(OPD,channel,eeta,t))
  Minv = (mp.eye(2*N)-OPK)**-1
  BIGM = -(2.0*t-1.0)*Minv*OPKdiff-t*(t-1.0)*(Minv*OPKdiff*Minv*OPKdiff\
         +Minv*OPKdiff2)
  value = 0.0
  for i in range(2*N) :
    value += BIGM[i,i]
  return (channel**2-thetas[0]**2-thetas[1]**2)/4.0-thetas[1]*thetas[2]/2.0+value

def accessoryKVI(OPA,OPD,thetas,channel,eeta,t,num=True):
  if num :
    return firstlogdiffVInum(OPA,OPD,thetas,channel,eeta,t)-thetas[1]*(thetas[0]/t+thetas[2]/(t-1))/2.0
  else :
    return firstlogdiffVI(OPA,OPD,thetas,channel,eeta,t)-thetas[1]*(thetas[0]/t+thetas[2]/(t-1))/2.0

def todaFVI(OPA,OPD,thetas,channel,eeta,t,num=True):
  if num :
    return secondlogdiffVInum(OPA,OPD,thetas,channel,eeta,t)+(thetas[1]-thetas[3])*thetas[1]/2.0
  else :
    return secondlogdiffVI(OPA,OPD,thetas,channel,eeta,t)+(thetas[1]-thetas[3])*thetas[1]/2.0

def firstlogdiffV(OPA,OPD,thetas,channel,t):
  def differential(x):
    return mp.log(tauV(OPA,OPD,thetas,channel,x))
  return mp.diff(differential,t)

def secondlogdiffV(OPA,OPD,thetas,channel,t):
  def differential(x):
    return t*firstlogdiffV(OPA,OPD,thetas,channel,x)
  return mp.diff(differential,t)

def accessoryKV(OPA,OPD,thetas,channel,t):
  return firstlogdiffV(coefficients,thetas,channel,t)+theta[0]*theta[1]

def todaFV(OPA,OPD,thetas,channel,t):
  return secondlogdiffV(coefficients,thetas,channel,t)+theta[1]/2.0
