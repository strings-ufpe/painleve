using ArbNumerics, LinearAlgebra

# This computes the Painlevé VI tau function using the Fourier decomposition outlined in
# GL2016.

global Nf = 48
setprecision(ArbComplex,digits=64)

function pieV(th1,th2,th3,sig)
    return (ArbNumerics.gamma(1-sig)^2*ArbNumerics.rgamma(1+sig)^2 *
            ArbNumerics.gamma(1+(th3+sig)/2)*ArbNumerics.rgamma(1+(th3-sig)/2) *
            ArbNumerics.gamma(1+(th2+th1+sig)/2)*ArbNumerics.rgamma(1+(th2+th1-sig)/2) *
            ArbNumerics.gamma(1+(th2-th1+sig)/2)*ArbNumerics.rgamma(1+(th2-th1-sig)/2))
end

function pieVI(th1,th2,th3,th4,sig)
    return (ArbNumerics.gamma(1-sig)^2*ArbNumerics.rgamma(1+sig)^2 *
            ArbNumerics.gamma(1+(th3+th4+sig)/2)*ArbNumerics.rgamma(1+(th3+th4-sig)/2) *
            ArbNumerics.gamma(1+(th3-th4+sig)/2)*ArbNumerics.rgamma(1+(th3-th4-sig)/2) *
            ArbNumerics.gamma(1+(th2+th1+sig)/2)*ArbNumerics.rgamma(1+(th2+th1-sig)/2) *
            ArbNumerics.gamma(1+(th2-th1+sig)/2)*ArbNumerics.rgamma(1+(th2-th1-sig)/2))
end

function ess(th1,th2,th3,th4,sig1,sig2,sig3)
    w1t = ArbNumerics.cospi(th2)*ArbNumerics.cospi(th3)+ArbNumerics.cospi(th1)*ArbNumerics.cospi(th4)
    w01 = ArbNumerics.cospi(th1)*ArbNumerics.cospi(th3)+ArbNumerics.cospi(th2)*cospi(th4)
    num = (w1t-ArbNumerics.cospi(sig2)-ArbNumerics.cospi(sig1)*ArbNumerics.cospi(sig3)) -
          (w01-ArbNumerics.cospi(sig3)-ArbNumerics.cospi(sig1)*ArbNumerics.cospi(sig2)) *
          ArbNumerics.exp(1im*pi*sig1)
    den = (ArbNumerics.cospi(th4)-ArbNumerics.cospi(th3-sig1)) *
          (ArbNumerics.cospi(th1)-ArbNumerics.cospi(th2-sig1))
    return num/den
end

function invseries(ser) #ser only being a 2x2 matrix vector
    inverse = zeros(ArbComplex, 2, 2, Nf+1);
    inverse[:,:,1] = [ 1 0 ; 0 1 ];
    for i = 2:(Nf+1)
        A = -ser[:,:,i];
        for j = 2:(i-1)
            A += - (ser[:,:,j]*inverse[:,:,(1+i-j)]);
        end
        inverse[:,:,i] = A
    end
    return inverse
end

function gee(sig,th1,th2,t0=1.0)
    psi = zeros(ArbComplex,2,2, Nf+1);
    # note sign of th1 to recover Nekrasov expansion
    a = (sig-th1+th2)/2;
    b = (sig-th1-th2)/2;
    c = sig
    psi[:,:,1] = [ 1 0 ; 0 1 ];
    psi[:,:,2] = [ ((a*b)/c*t0) (-a*b/c/(1+c)*t0) ;
                   ((a-c)*(b-c)/c/(1-c)*t0) ((a-c)*(b-c)/(-c)*t0) ];
    for p = 3:(Nf+1)
        psi[1,1,p] = ((a+p-2)*(b+p-2)/((p-1)*(c+p-2))*psi[1,1,p-1]*t0)
        psi[1,2,p] = ((a+p-2)*(b+p-2)/((p-2)*(c+p-1))*psi[1,2,p-1]*t0)
        psi[2,1,p] = ((a-c+p-2)*(b-c+p-2)/((p-2)*(-c+p-1))*psi[2,1,p-1]*t0)
        psi[2,2,p] = ((a-c+p-2)*(b-c+p-2)/((p-1)*(-c+p-2))*psi[2,2,p-1]*t0)
    end
    return psi
end

function geeV(th1,th2,t0=1.0)
    psi = zeros(ArbComplex,2,2,Nf+1)
    # Different definition to LNR2018
    a = (th1-th2)/2
    c = th1
    psi[:,:,1] = [ 1 0 ; 0 1 ]
    psi[:,:,2] = [ (a/c*t0) (-a/(c*(1+c))*t0) ; ((a-c)/(c*(1-c))*t0) (-(a-c)/c*t0) ]
    for p = 3:Nf+1
        psi[1,1,p] = ((a+p-2)/((c+p-2)*(p-1))*psi[1,1,p-1]*t0)
        psi[1,2,p] = ((a+p-2)/((c+p-1)*(p-2))*psi[1,2,p-1]*t0)
        psi[2,1,p] = ((a-c+p-2)/((-c+p-1)*(p-2))*psi[2,1,p-1]*t0)
        psi[2,2,p] = ((a-c+p-2)/((-c+p-2)*(p-1))*psi[2,2,p-1]*t0)
    end
    return psi
end

function BuildA(sig,th1,th2)
    vecg = gee(sig,th1,th2)
    vecginv = invseries(vecg)
    bigA = ArbNumerics.ArbComplexMatrix(zeros(ArbComplex,2Nf,2Nf))
    for p = 1:Nf
        for q = 1:Nf
            result = zeros(ArbComplex,2,2)
            if q + p <= Nf+1
                for r = 1:q
                    result += vecg[:,:,p+r]*vecginv[:,:,q-r+1]
                end
            end
            bigA[(2p-1):(2p), (2q-1):(2q)] = result
        end
    end
    return bigA
end

function BuildDVI(sig,th1,th2,x,t)
    vecg = gee(sig,th1,th2,t);
    vecginv = invseries(vecg)
    bigD = ArbNumerics.ArbComplexMatrix(zeros(ArbComplex,2Nf,2Nf))
    left = [ (1/x) 0 ; 0 1 ]
    right = [ (x) 0 ; 0 1 ]
    for p = 1:Nf
        for q = 1:Nf
            result = zeros(ArbComplex,2,2)
            if q + p <= Nf+1
                for r = 1:p
                    result += -vecg[:,:,p-r+1]*vecginv[:,:,q+r]
                end
            end
            bigD[(2p-1):(2p), (2q-1):(2q)] = left*result*right
        end
    end
    return bigD
end

function BuildDV(sig,ths,x,t0)
    vecg = geeV(sig,ths,t0)
    vecginv = invseries(vecg)
    bigD = ArbNumerics.ArbComplexMatrix(zeros(ArbComplex,2Nf,2Nf))
    left = [ (1/x) 0 ; 0 1 ]
    right = [ (x) 0 ; 0 1 ]
    for p = 1:Nf
        for q = 1:Nf
            result = zeros(ArbComplex,2,2)
            if q + p <= Nf+1
                for r = 1:p
                    result += -vecg[:,:,p-r+1]*vecginv[:,:,q+r]
                end
            end
            bigD[2*p-1:2*p,2*q-1:2*q] = left*result*right
        end
    end
    return bigD
end

function BuildDVIdiff(sig,th1,th2,x,t)
    vecg = gee(sig,th1,th2,t)
    vecginv = invseries(vecg)
    bigD = ArbNumerics.ArbComplexMatrix(zeros(ArbComplex,2Nf,2Nf))
    left = [ (1/x) 0 ; 0 1 ]
    right = [ (x) 0 ; 0 1 ]
    sigma3 = [ 1 0 ; 0 -1 ]
    for p = 1:Nf
        for q = 1:Nf
            result = zeros(ArbComplex,2,2)
            if q + p <= Nf+1
                for r = 1:p
                    result += -vecg[:,:,p-r+1]*vecginv[:,:,q+r]
                end
            end
            result = (p+q-1)/t*result+sig/(2t)*(sigma3*result-result*sigma3)
            bigD[(2p-1):(2p), (2q-1):(2q)] = left*result*right
            end
    end
    return bigD
end

function BuildDVdiff(sig,ths,x,t)
    vecg = geeV(sig,ths,t)
    vecginv = invseries(vecg)
    bigD = ArbNumerics.ArbComplexMatrix(zeros(ArbComplex,2Nf,2Nf))
    left = [ (1/x) 0 ; 0 1 ]
    right = [ (x) 0 ; 0 1 ]
    sigma3 = [ 1 0 ; 0 -1 ]
    for p = 1:Nf
        for q = 1:Nf
            result = zeros(ArbComplex,2,2)
            if q + p <= Nf+1
                for r = 1:p
                    result += -vecg[:,:,p-r+1]*vecginv[:,:,q+r]
                end
            end
            result = (p+q-1)/t*result+sig/(2t)*(sigma3*result-result*sigma3)
            bigD[(2p-1):(2p), (2q-1):(2q)] = left*result*right
            end
    end
    return bigD
end

function tauhatVI(th,sig,esse,t)
    x = esse*pieVI(th[1],th[2],th[3],th[4],sig)*t^sig
    OpA = BuildA(sig,th[3],th[4])
    OpD = BuildDVI(-sig,th[2],th[1],x,t)
    id = Matrix{ArbComplex}(I,2Nf,2Nf)
    Fred = ArbNumerics.ArbComplexMatrix(id-OpD*OpA)
    return (ArbNumerics.determinant(Fred))
end

function tauhatV(th,sig,esse,t)
    # theta[1] = theta1, theta[2] = thetainfinity, theta[3] = thetastar
    x = esse*pieV(th[1],th[2],th[3],sig)*t^sig
    OPA = BuildA(sig,th[2],th[1])
    OPD = BuildDV(-sig,th[3],x,-t)
    id = Matrix{ArbComplex}(I,2Nf,2Nf)
    Fred = ArbNumerics.ArbComplexMatrix(id - OPA*OPD)
    return ArbNumerics.determinant(Fred)
end

function firstlogdiffVI(th,sig,esse,t)
    x = esse*pieVI(th[1],th[2],th[3],th[4],sig)*t^sig
    OpA = BuildA(sig,th[3],th[4])
    OpD = BuildDVI(-sig,th[2],th[1],x,t)
    OpDdiff = BuildDVIdiff(-sig,th[2],th[1],x,t)
    Id = Matrix{ArbComplex}(I,2Nf,2Nf)
    OpKi = (ArbNumerics.ArbComplexMatrix(Id-OpA*OpD))^(-1)
    OpKdiff = ArbNumerics.ArbComplexMatrix(OpA*OpDdiff)
    return -ArbNumerics.tr(ArbNumerics.ArbComplexMatrix(OpKdiff*OpKi))
end

function firstlogdiffV(th,sig,esse,t)
    x = esse*pieV(th[1],th[2],th[3],sig)*t^sig
    OpA = BuildA(sig,th[2],th[1])
    OpD = BuildDV(-sig,th[3],x,-t)
    OpDdiff = BuildDVdiff(-sig,th[3],x,-t)
    Id = Matrix{ArbComplex}(I,2Nf,2Nf)
    OpKi = (ArbNumerics.ArbComplexMatrix(Id-OpA*OpD))^(-1)
    OpKdiff = ArbNumerics.ArbComplexMatrix(OpA*OpDdiff)
    return ArbNumerics.tr(ArbNumerics.ArbComplexMatrix(OpKdiff*OpKi))
end

function tauVI(th,sig,esse,t)
    prefactor = t^((sig^2-th[1]^2-th[2]^2)/4)*(1-t)^(-th[2]*th[3]/2)
    return prefactor*tauhatVI(th,sig,esse,t)
end

function tauV(th,sig,esse,t,GIL=false)
    # this is different from GIL2013:
    # esse{here} = -esse{GIL}*exp(+im*pi*sigma)
    # th[3]{here} = -th[3]{GIL}
    # t{here} = -t{GIL}
    if GIL
        thGIL = copy(th)
        thGIL[3] = -th[3]
        esseGIL = -esse*ArbNumerics.exp(-ArbComplex(pi*im)*sig)
        return t^(sig^2/4)*tauhatV(thGIL,sig,esseGIL,-t)
    else
        prefactor = t^((sig^2-th[3]^2/2)/4)*ArbNumerics.exp(-th[2]/2*t)
        return prefactor*tauhatV(th,sig,esse,t)
    end
end

function accessoryKVI(th,sig,esse,t)
    # This is the accessory parameter K defined by (4.1) in ACCN2018
    # this implements the Schlesinger move in ACCN2018
    theta = copy(th)
    theta[2] -= 1
    theta[4] += 1
    sig -= 1
    # esse is invariant
    prefactor = (sig^2-(theta[1]+theta[2])^2)/(4t)-theta[2]*theta[3]/(t-1)
    return prefactor+firstlogdiffVI(theta,sig,esse,t)
end

function accessoryKV(th,sig,esse,t)
    theta = copy(th)
    theta[3] -= 1
    theta[2] -= 1 # this sign needs checking
    sig -= 1
    prefactor = (sig^2-theta[3]^2)/(4t)+theta[2]
    return prefactor+firstlogdiffV(theta,sig,esse,t)
end

function hirotaVI(th,sig,esse,t)
    # this implements the sigma_{VI} function as in GIL2013
    prefactor = (t-1)*(2*sig^2-th[1]^2-th[2]^2-th[3]^2-th[4]^2)/8 -
                t*(th[1]^2-th[2]^2+th[3]^2-th[4]^2+4*th[2]*th[3])/8
    return t*(t-1)*firstlogdiffVI(th,sig,esse,t)+prefactor
end

function hirotaV(th,sig,esse,t)
    prefactor = (sig^2-th[1]^2-th[2]^2-th[3]^2/2)/4-th[3]*t/4
    return t*firstlogdiffV(th,sig,esse,t)+prefactor
end
