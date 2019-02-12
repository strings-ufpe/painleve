#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sympy import *
from mpmath import mp
import matplotlib.pyplot as plt
import numpy as np

t = symbols('t')
#th0, tht, th1, thi = symbols('\\theta_0 \\theta_t \\theta_1 \\theta_\infty')
#sig, s1t, s01, ess = symbols('\sigma \sigma_{1t} \sigma_{01} \\tilde{s}')
#tau, cvi, cbl = symbols('\\tau C_{VI} cbl')
th0, tht, thi = symbols('theta0 thetat thetai')
sig, s1t, s01, ess = symbols('sigma s1t s01 ess')
tau, cvi, cbl = symbols('tau cvi cbl')

Nexp = 4
Nblk = 10
'The number Nexp is associated to the channels n, in the future I will fix Nexp =3'
'The number Nblk is associated to the expansion of the Conformal Block, in the future I will fix Nblk =8'

'Building the Young Tableaux'
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

'Building the HookLenght - h_{i,j}'

def hook_length(partition,i,j):
  if len(partition) > 0:
    trans = transpose(partition)
    return partition[i]+trans[j]-i-j-1
  else:
    return 0

'Building the Nekrasov partitions'

def coefficientV(partition1,partition2,channel):
  b12 = 1
  for i1 in range(len(partition1)):
    for k1 in range(partition1[i1]):
      b12 *= ((tht+channel+2*i1-2*k1)**2-th0**2)\
             *((-thi+channel+2*i1-2*k1))\
             /(8*hook_length(partition1,i1,k1)**2\
             *((setarm(transpose(partition1),k1)+setarm(partition2,i1)\
             -i1-k1-1)+channel)**2)
  for i2 in range(len(partition2)):
    for k2 in range(partition2[i2]):
      b12 *= ((tht-channel+2*i2-2*k2)**2-th0**2)\
             *((-thi-channel+2*i2-2*k2))\
             /(8*hook_length(partition2,i2,k2)**2\
             *((setarm(transpose(partition2),k2)+setarm(partition1,i2)\
             -i2-k2-1)-channel)**2)
  return b12
'Building the Conformal Block Pv'
def conformal_blockV(channel,level):
  cf12 = 0
  p = level
  q = 0
  while p >= 0:
    for partition1 in list(partitions(p)):
      for partition2 in list(partitions(q)):
        cf12 += coefficientV(partition1[::-1],partition2[::-1],channel)
    p -= 1
    q += 1
  return S(cf12)

def pochhammer(arg,level):
  res = S(1)
  if level > 0:
    while level > 0:
      res *= arg+level-1
      level -= 1
  else:
    while level < 0:
      res /= arg+level
      level += 1
  return res
'Here I used the result that I have from Accessory Parameter'
def structure_constantV(channel,level):
  res = S(1)
  if level > 0:
    while level > 0:
      level -= 1
      res *= -((-thi-sig-2*level))*((tht-sig-2*level)**2-th0**2)\
             /((8*(sig+2*level)**2*(1+sig+2*level)**2))\
             *(pochhammer(1-sig,-2*level)/pochhammer(1+sig,2*level))**2\
             *(pochhammer(1+(-thi+sig)/2,level)/pochhammer(1+(-thi-sig)/2,-level))\
             *(pochhammer(1+(tht+th0+sig)/2,level)/pochhammer(1+(tht+th0-sig)/2,-level))\
             *(pochhammer(1+(tht-th0+sig)/2,level)/pochhammer(1+(tht-th0-sig)/2,-level))
  else:
    while level < 0:
      level += 1
      res *= -((-thi+sig+2*level))*((tht+sig+2*level)**2-th0**2)\
             /((8*(sig+2*level)**2*(1-sig-2*level)**2))\
             *(pochhammer(1+sig,2*level)/pochhammer(1-sig,-2*level))**2\
             *(pochhammer(1+(-thi-sig)/2,-level)/pochhammer(1+(-thi+sig)/2,level))\
             *(pochhammer(1+(tht+th0-sig)/2,-level)/pochhammer(1+(tht+th0+sig)/2,level))\
             *(pochhammer(1+(tht-th0-sig)/2,-level)/pochhammer(1+(tht-th0+sig)/2,level))
  return res

def tauexpansion(arg):
  tau = S(0)
  for een in range(2*Nexp+1):
    partial = S(0)
    for eem in range(Nblk):
      partial += conformal_blockV(sig+2*(een-Nexp),eem)*t**(eem)
    tau += structure_constantV(sig,een-Nexp)*arg**(een-Nexp)*t**((een-Nexp)*(een-Nexp))*partial
  return tau

'The script now has the Painleve V expansion'
def essratio():
    kappa = (gamma(1-sig)**2/gamma(1+sig)**2)\
            *gamma(1+(tht+th0+sig)/2)*gamma(1+(tht-th0+sig)/2)\
            *gamma(1+(-thi+sig)/2)\
            /(gamma(1+(tht+th0-sig)/2)*gamma(1+(tht-th0-sig)/2)\
            *gamma(1+(-thi-sig)/2))
    kappam = kappa.subs([(tht,tht-1),(thi,thi+1),(sig,sig-1)])
    return simplify(kappam/kappa)

ci = list(symbols('ci:{}'.format(Nblk*(2*Nexp+1))))
def taumock(arg):
  tau = S(0)
  for een in range(2*Nexp+1):
    for eem in range(0,Nblk):
      tau += ci[een*Nblk+eem]*arg**(een-Nexp)*t**((een-Nexp)*(een-Nexp)+eem)
  return tau

cm = list(symbols('cm:{}'.format(Nblk*(2*Nexp+1))))
def taumockm(arg):
  tau = S(0)
  for een in range(2*Nexp+1):
    for eem in range(0,Nblk):
      tau += cm[een*Nblk+eem]*arg**(een-Nexp)*t**((een-Nexp)*(een-Nexp)+eem)
  return tau

def computekappa(dee,positive=True):

  arg, ess0, x0 = symbols('arg ess0 x0')
  tau = taumock(arg)
  c1f = list(symbols('c1f:{}'.format(dee+1)))
  c2f = list(symbols('c2f:{}'.format(dee+1)))
  c1f[0] = 1
  c2f[0] = 1
  if positive:
    ess = ess0*t*sum(c1f[n]*t**n for n in range(dee))
  else:
    ess = 1/(ess0*t)*sum(c1f[n]*t**n for n in range(dee))
  equation = tau.subs(arg,ess)
  x0 = solve(equation.subs(t,0),ess0)[0]
  equation = equation.subs(ess0,x0)
  for k in range(1,dee):
    equation = diff(equation.subs(c1f[k-1],c2f[k-1]),t)
    c2f[k] = solve(equation.subs(t,0),c1f[k])[0]
  if positive:
    sol = x0*t*(sum(c2f[n]*t**n for n in range(dee)))
  else:
    sol = 1/(x0*t)*(sum(c2f[n]*t**n for n in range(dee)))
  #sol = sol.subs(ess0,x0)
  ck = sol.subs(t,1).free_symbols
  for een in range(2*Nexp+1):
    for eem in range(Nblk):
      if ci[een*Nblk+eem] in ck:
        var = structure_constantV(sig,een-Nexp)*conformal_blockV(sig+2*(een-Nexp),eem)
        sol = sol.subs(ci[een*Nblk+eem],var)
  #print(sol.subs(t,1).free_symbols)
  return(sol)

def kappamockminus(dee,positive=True):
  arg, ess0, x0 = symbols('arg ess0 x0')
  tau = taumockm(arg)
  c1f = symarray('c1f',dee+1)
  c2f = symarray('c2f',dee+1)
  c1f[0] = S(1)
  c2f[0] = S(1)
  if positive:
    ess = ess0*t*sum(c1f[n]*t**n for n in range(dee))
  else:
    ess = 1/(ess0*t)*sum(c1f[n]*t**n for n in range(dee))
  equation = tau.subs(arg,ess)
  x0 = solve(equation.subs(t,0),ess0)[0]
  equation = equation.subs(ess0,x0)
  for k in range(1,dee):
    equation = diff(equation.subs(c1f[k-1],c2f[k-1]),t)
    c2f[k] = solve(equation.subs(t,0),c1f[k])[0]
  if positive:
    sol = sum(c2f[n]*t**n for n in range(dee))
  else:
    sol = sum(c2f[n]*t**n for n in range(dee))
  return(sol)

def invertseries(exp,nmax):
  c1f = list(symbols('c1f:{}'.format(nmax+1)))
  c2f = list(symbols('c2f:{}'.format(nmax+1)))
  inv = sum(c1f[n]*t**n for n in range(nmax+1))
  equation = inv*exp-1
  for k in range(nmax+1):
    c2f[k] = solve(equation.subs(t,0),c1f[k])[0]
    equation = diff(equation.subs(c1f[k],c2f[k]),t)/(k+1)
  sol = sum(c2f[n]*t**n for n in range(nmax+1))
  return sol

def truncate(exp,dee):
  res = S(0)
  for k in range(dee+1):
    res += exp.subs(t,0)*t**k
    exp = diff(exp,t)/(k+1)
  return res

def difflogtauhat(dee,positive=True):
  arg, ess = symbols('arg ess')
  tau = taumock(arg).subs(arg,ess*t**sig)
  kappa = kappamockminus(dee+1)
  num = diff(tau,t).subs(ess, kappa*t**(-sig))
  den = invertseries(tau.subs(ess,kappa*t**(-sig)),dee)
  equation = truncate((num*den),dee)
  ck = equation.subs(t,1).free_symbols
  struct = symarray('',2*Nexp+1)
  conf = symarray('',Nblk)
  for een in range(2*Nexp+1):
    struct[een] = structure_constantV(sig,een-Nexp)
  for eem in range(dee+2):
    conf[eem] = conformal_blockV(sig,eem)
  for een in range(2*Nexp+1):
    for eem in range(Nblk):
       if ci[een*Nblk+eem] in ck:
         var = struct[een]*((conf[eem]).subs(sig,sig+2*(een-Nexp)))
         equation = equation.subs(ci[een*Nblk+eem],var)
  equation = equation.subs([(tht,tht-1),(thi,thi+1),(sig,sig-1)])
  for een in range(2*Nexp+1):
    for eem in range(Nblk):
      if cm[een*Nblk+eem] in ck:
        var = struct[een]*((conf[eem]).subs(sig,sig+2*(een-Nexp)))
        equation = equation.subs(cm[een*Nblk+eem],var)
  return equation

def accessoryK(dee,positive=True):
  return (((sig-1)**2)/(4*t)-(tht-1)/2-((th0+tht-1)**2)/(4*t))+difflogtauhat(dee,positive)

def computesigma(teekey,dee):
  x = symbols('x')
  equation = (x-(th0+tht-1)**2)+4*t*difflogtauhat(dee).subs((sig-1)**2,x)-teekey
  s1f = list(symbols('s1f:{}'.format(dee+1)))
  s2f = list(symbols('s2f:{}'.format(dee+1)))
  exp = sum(s1f[n]*t**n for n in range(dee+1))
  equation = equation.subs(sig,exp)
  for k in range(dee+1):
    print(equation)
    s2f[k]=solve(equation.subs(t,0),s1f[k])[0]
    equation = diff(equation.subs(s1f[k],s2f[k]),t)/(k+1)
  sol = sum(s2f[n]*t**n for n in range(dee+1))
  return sol

def difflogtau(dee):
  arg, ess, kap = symbols('arg ess kappa')
  tau = taumock(arg).subs(arg,ess*t**sig)
  num = diff(tau,t).subs(ess, arg*t**(-sig))
  den = invertseries(tau.subs(ess,arg*t**(-sig)),dee)
  equation = num*den
  ck = equation.subs(t,1).free_symbols
  for een in range(2*Nexp+1):
    struct[een] = structure_constantV(sig,een-Nexp)
    for eem in range(Nblk):
      if ci[een*Nblk+eem] in ck:
        var = struct[een]*conformal_blockV(sig+2*(een-Nexp),eem)
        equation = equation.subs(ci[een*Nblk+eem],var)
  return equation


'''
thetas = [ 0.3804+0.6212j, 0.8364-0.4218j, 0.6858+0.6628j, 0.0326+0.3610j ]
sigmas = [ -0.6544-0.9622j, 0.1916+0.6336j, 0.9524+0.2056j ]
channel = -0.6544-0.9622j
ess = 0.1916+0.6336j

kappa = (gamma(1-sig)**2/gamma(1+sig)**2)\
        *gamma(1+(tht+th0+sig)/2)*gamma(1+(tht-th0+sig)/2)\
        *gamma(1+(-thi+sig)/2)\
        /(gamma(1+(tht+th0-sig)/2)*gamma(1+(tht-th0-sig)/2)\
        *gamma(1+(-thi-sig)/2))*ess
'''
