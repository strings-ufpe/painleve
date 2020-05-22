using Painleve, ArbNumerics, LinearAlgebra

println("these values are from GIL2013, root computed via Nekrasov expansion")

global theta = [ ArbComplex(0.3804+0.6212im) ArbComplex(0.8364-0.4218im) ArbComplex(0.6858+0.6628im) ArbComplex(0.0326+0.3610im) ]
global sigma = [ ArbComplex(-0.6544-0.9622im) ArbComplex(0.1916+0.6336im) ArbComplex(0.9524+0.2056im) ]
esse = ess(theta[1],theta[2],theta[3],theta[4],sigma[1],sigma[2],sigma[3])
#t0 = Complex{BigFloat}(0.30834919880489809653313980211353743299530186116517 + 0.19086743009424488900720970772011099048904038492822im)
t0 = ArbComplex(0.3083364603615607914977066514996857457511+0.190867262128723524107182362204876464236im)
c0 = ArbComplex(-1.6071-0.10556im)

value = tauhatVI(theta,sigma[1],esse,t0)
println("this is the tau function, should print close to zero")
println(value)
println("This should return K0 = -0.7598203871514175 - 0.29233776128695677im")
value = accessoryKVI(theta,sigma[1],esse,t0)
println(value)

println("")
println("these values are from ACCN2018.")

global theta = [ ArbComplex(0.1827991846) ArbComplex(0.2869823004) ArbComplex(0.3673544015) ArbComplex(0.0853271421) ]
global sigma = [ ArbComplex(1-0.4304546489im) ArbComplex(1-0.5385684561im) ArbComplex(0.9631297769+0.7221017400im) ]
esse = ess(theta[1],theta[2],theta[3],theta[4],sigma[1],sigma[2],sigma[3])
t0 = ArbComplex(0.2086468690)
c0 = ArbComplex(-0.4364792362)

value = tauhatVI(theta,sigma[1],esse,t0)
println("this is the tau function, should print close to zero")
println(value)
println("this should return K0 = -0.4364792365658455 (last digit in ACCN2018 is wrong)")
value = accessoryKVI(theta,sigma[1],esse,t0)
println(value)

println("")
println("Values below are from the radial equation of Kerr. For a=0.96")
println("No. of sig. digits is significantly lower for PV")
setprecision(ArbComplex,digits=128)
global theta = [ ArbComplex(-2.386360765707078 - 2.15459461970831im) ArbComplex(-1.3131364165207502 + 3.830390435036997im) ArbComplex(-4.300502817772172 - 1.6757958153286867im) ]
global sigma = ArbComplex(0.7622014124175833 - 0.4762815331078994im)
esse = ArbComplex(0.003841579720247019 + 0.050011279822625564im)
t0 = ArbComplex(-0.08414078897620816 - 0.4692228282920325im)

# these should return -2.32734255074356e-12 - 4.1567581505361e-13im
value = tauhatV(theta,sigma,esse,t0)
println("this is the tau function, should print close to zero")
println(value)
println("this is the accessory parameter, K0 = 6.4575940358 + 0.163454096im")
value = accessoryKV(theta,sigma,esse,t0)
println(value)
println("Done")
