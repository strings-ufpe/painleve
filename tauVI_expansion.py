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
th0, tht, th1, thi = symbols('theta0 thetat theta1 thetai')
sig, s1t, s01, ess = symbols('sigma s1t s01 ess')
tau, cvi, cbl = symbols('tau cvi cbl')

Nexp = 3
Nblk = 8

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

def hook_length(partition,i,j):
  if len(partition) > 0:
    trans = transpose(partition)
    return partition[i]+trans[j]-i-j-1
  else:
    return 0

def coefficientVI(partition1,partition2,channel):
  b12 = 1
  for i1 in range(len(partition1)):
    for k1 in range(partition1[i1]):
      b12 *= ((tht+channel+2*i1-2*k1)**2-th0**2)\
             *((th1+channel+2*i1-2*k1)**2-thi**2)\
             /(16*hook_length(partition1,i1,k1)**2\
             *((setarm(transpose(partition1),k1)+setarm(partition2,i1)\
             -i1-k1-1)+channel)**2)
  for i2 in range(len(partition2)):
    for k2 in range(partition2[i2]):
      b12 *= ((tht-channel+2*i2-2*k2)**2-th0**2)\
             *((th1-channel+2*i2-2*k2)**2-thi**2)\
             /(16*hook_length(partition2,i2,k2)**2\
             *((setarm(transpose(partition2),k2)+setarm(partition1,i2)\
             -i2-k2-1)-channel)**2)
  return b12

def conformal_blockVI(channel,level):
  cf12 = 0
  p = level
  q = 0
  while p >= 0:
    for partition1 in list(partitions(p)):
      for partition2 in list(partitions(q)):
        cf12 += coefficientVI(partition1[::-1],partition2[::-1],channel)
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

def structure_constantVI(channel,level):
  res = S(1)
  if level > 0:
    while level > 0:
      level -= 1
      res *= -((th1-sig-2*level)**2-thi**2)*((tht-sig-2*level)**2-th0**2)\
             /((16*(sig+2*level)**2*(1+sig+2*level)**2))\
             *(pochhammer(1-sig,-2*level)/pochhammer(1+sig,2*level))**2\
             *(pochhammer(1+(th1+thi+sig)/2,level)/pochhammer(1+(th1+thi-sig)/2,-level))\
             *(pochhammer(1+(th1-thi+sig)/2,level)/pochhammer(1+(th1-thi-sig)/2,-level))\
             *(pochhammer(1+(tht+th0+sig)/2,level)/pochhammer(1+(tht+th0-sig)/2,-level))\
             *(pochhammer(1+(tht-th0+sig)/2,level)/pochhammer(1+(tht-th0-sig)/2,-level))
  else:
    while level < 0:
      level += 1
      res *= -((th1+sig+2*level)**2-thi**2)*((tht+sig+2*level)**2-th0**2)\
             /((16*(sig+2*level)**2*(1-sig-2*level)**2))\
             *(pochhammer(1+sig,2*level)/pochhammer(1-sig,-2*level))**2\
             *(pochhammer(1+(th1+thi-sig)/2,-level)/pochhammer(1+(th1+thi+sig)/2,level))\
             *(pochhammer(1+(th1-thi-sig)/2,-level)/pochhammer(1+(th1-thi+sig)/2,level))\
             *(pochhammer(1+(tht+th0-sig)/2,-level)/pochhammer(1+(tht+th0+sig)/2,level))\
             *(pochhammer(1+(tht-th0-sig)/2,-level)/pochhammer(1+(tht-th0+sig)/2,level))
  return res

def tauexpansion(arg):
  tau = S(0)
  for een in range(2*Nexp+1):
    partial = S(0)
    for eem in range(Nblk):
      partial += conformal_blockVI(sig+2*(een-Nexp),eem)*t**(eem)
    tau += structure_constantVI(sig,een-Nexp)*arg**(een-Nexp)*t**((een-Nexp)*(een-Nexp))*partial
  return tau

def essratio():
  p0 = 2*cos(pi*th0)
  pt = 2*cos(pi*tht)
  p1 = 2*cos(pi*th1)
  pf = 2*cos(pi*thi)
  p0t = 2*cos(pi*sig)
  p1t = 2*cos(pi*s1t)
  p01 = 2*cos(pi*s01)
  eeta = ((pt*p1+p0*pf-2*p1t-p0t*p01)-(p0*p1+pt*pf-2*p01-p0t*p1t)*exp(I*pi*sig))\
         /((2*cos(pi*(tht-sig))-p0)*(2*cos(pi*(th1-sig))-pf))
  kappa = (gamma(1-sig)**2/gamma(1+sig)**2)\
          *gamma(1+(tht+th0+sig)/2)*gamma(1+(tht-th0+sig)/2)\
          *gamma(1+(th1+thi+sig)/2)*gamma(1+(th1-thi+sig)/2)\
          /(gamma(1+(tht+th0-sig)/2)*gamma(1+(tht-th0-sig)/2)\
          *gamma(1+(th1+thi-sig)/2)*gamma(1+(th1-thi-sig)/2))
  eetam = eeta.subs([(tht,tht-1),(thi,thi+1),(sig,sig-1),(s1t,s1t-1)])
  kappam = kappa.subs([(tht,tht-1),(thi,thi+1),(sig,sig-1),(s1t,s1t-1)])
  return simplify(kappam/kappa*eetam/eeta)

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
        var = structure_constantVI(sig,een-Nexp)*conformal_blockVI(sig+2*(een-Nexp),eem)
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
    struct[een] = structure_constantVI(sig,een-Nexp)
  for eem in range(Nblk):
    conf[eem] = conformal_blockVI(sig,eem)
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

'''
def difflogtauhat(dee,positive=True):
  arg, ess = symbols('arg ess')
  tau = taumock(arg).subs(arg,ess*t**sig)
  num = diff(tau,t).subs(ess, arg*t**(-sig))
  den = invertseries(tau.subs(ess,arg*t**(-sig)),dee)
  equation = num*den
  ck = equation.subs(t,1).free_symbols
  for een in range(2*Nexp+1):
    for eem in range(Nblk):
      if ci[een*Nblk+eem] in ck:
        var = structure_constantVI(sig,een-Nexp)*conformal_blockVI(sig+2*(een-Nexp),eem)
        equation = equation.subs(ci[een*Nblk+eem],var)
  equation = equation.subs([(tht,tht-1),(thi,thi+1),(sig,sig-1)])
  essm = computekappa(dee+1,positive)*essratio()/t
  equation = equation.subs(arg,essm)
  res = S(0)
  for k in range(dee+1):
    res += equation.subs(t,0)*t**k
    equation = diff(equation,t)/(k+1)
  return res
'''

def accessoryK(dee,positive=True):
  return ((sig-1)**2-(th0+tht-1)**2)/(4*t)+difflogtauhat(dee,positive)

def computesigma(fourteekey,dee):
  x = symbols('x')
  equation = (x-(th0+tht-1)**2)+4*t*difflogtauhat(dee).subs((sig-1)**2,x)-fourteekey
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


'''
thetas = [ 0.3804+0.6212j, 0.8364-0.4218j, 0.6858+0.6628j, 0.0326+0.3610j ]
sigmas = [ -0.6544-0.9622j, 0.1916+0.6336j, 0.9524+0.2056j ]
channel = -0.6544-0.9622j

pis = [ 2.0*mp.cos(mp.pi*thetas[i1]) for i1 in range(len(thetas)) ]
pijs = [ 2.0*mp.cos(mp.pi*sigmas[i1]) for i1 in range(len(sigmas)) ]

eess = ((pis[1]*pis[2]+pis[0]*pis[3]-2.0*pijs[1]-pijs[0]*pijs[2])\
       -(pis[0]*pis[2]+pis[1]*pis[3]-2.0*pijs[2]-pijs[0]*pijs[1])\
       *mp.exp(mp.pi*1j*sigmas[0]))\
       /(2*mp.cos(mp.pi*(thetas[1]-sigmas[0]))-pis[0])\
       /(2*mp.cos(mp.pi*(thetas[2]-sigmas[0]))-pis[3])
sigma = sigmas[0]
eeta = eess*(mp.gamma(1-sigma)**2/mp.gamma(1+sigma)**2)\
       *mp.gamma(1+(thetas[1]+thetas[0]+sigma)/2)*mp.gamma(1+(thetas[1]-thetas[0]+sigma)/2)\
       *mp.gamma(1+(thetas[2]+thetas[3]+sigma)/2)*mp.gamma(1+(thetas[2]-thetas[3]+sigma)/2)\
       /(mp.gamma(1+(thetas[1]+thetas[0]-sigma)/2)*mp.gamma(1+(thetas[1]-thetas[0]-sigma)/2)\
       *mp.gamma(1+(thetas[2]+thetas[3]-sigma)/2)*mp.gamma(1+(thetas[2]-thetas[3]-sigma)/2))

def structure_constantN(theta,channel):

  cn12 = 1.0+0.0j
  for i1 in ( mp.mpc('-1.0'), mp.mpc('1.0') ):
    for j1 in ( mp.mpc('-1.0'), mp.mpc('1.0') ):
      cn12 *= mp.barnesg(1+theta[1]/2.0+j1*theta[0]/2.0+i1*channel/2.0)\
              *mp.barnesg(1+theta[2]/2.0+j1*theta[3]/2.0+i1*channel/2.0)
    cn12 /= mp.barnesg(1+i1*channel)
  return cn12

def sigmaVI(arg):
  tauhat = lambdify([th0,tht,th1,thi,sig,ess,t],tau,"numpy")
  difftauhat = lambdify([th0,tht,th1,thi,sig,ess,t],difftau,"numpy")
  res = difftauhat(thetas[0],thetas[1],thetas[2],thetas[3],channel,eeta,arg)\
        /tauhat(thetas[0],thetas[1],thetas[2],thetas[3],channel,eeta,arg)\
        +(channel**2-thetas[0]**2-thetas[1]**2)/(4*arg)\
        +thetas[1]*thetas[2]/(2*(1-arg))+(thetas[0]**2+thetas[1]**2-thetas[2]**2-thetas[3]**2)/(8*arg)\
        +(thetas[1]**2+thetas[2]**2-thetas[0]**2-thetas[3]**2)/(8*(arg-1.0))
  return res*arg*(arg-1.0)

test0 = mp.convert('(0.3083492 + 0.1908674j)')
tauhat = lambdify([th0,tht,th1,thi,sig,ess,t],tau,"numpy")
print(tauhat(thetas[0],thetas[1],thetas[2],thetas[3],channel,eeta,test0))

times = mp.linspace(0.01,0.9,40,endpoint=False)
results = [ sigmaVI(t) for t in times ]
fig = plt.figure(frameon=False)
fig.set_size_inches(6.0,4.8)
graph = np.array(results,dtype=np.complex128)
plt.plot(times,graph.real)
plt.plot(times,graph.imag)
plt.grid()
plt.show()

thetas = [ mp.convert(thetas[i1]) for i1 in range(len(thetas)) ]
channel = mp.convert(channel)
kappa = (mp.gamma(1-channel)**2/mp.gamma(1+channel)**2)\
       *mp.gamma(1+(thetas[1]+thetas[0]+channel)/2)*mp.gamma(1+(thetas[1]-thetas[0]+channel)/2)\
       *mp.gamma(1+(thetas[2]+thetas[3]+channel)/2)*mp.gamma(1+(thetas[2]-thetas[3]+channel)/2)\
       /(mp.gamma(1+(thetas[1]+thetas[0]-channel)/2)*mp.gamma(1+(thetas[1]-thetas[0]-channel)/2)\
       *mp.gamma(1+(thetas[2]+thetas[3]-channel)/2)*mp.gamma(1+(thetas[2]-thetas[3]-channel)/2))
'''
