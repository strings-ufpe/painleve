using LinearAlgebra
using Base.Threads
using Nemo

# This implments the Fredholm determinant expansion for the Painlevé V
# tau function, numerically. It is written in arbitrary precision routine

JULIA_NUM_THREADS=4
N = 24
CC = ComplexField(128)
S = MatrixSpace(CC,2N,2N)

function computeη(θμ, σ, ess)
    # value ess is s_V from GIL
    Γfac = (gamma(1-σ)^2*rgamma(1+σ)^2)*
                gamma(1-(θμ[3]-σ)/2)*rgamma(1-(θμ[3]+σ)/2)*
                gamma(1+(θμ[2]+θμ[1]+σ)/2)*rgamma(1+(θμ[2]+θμ[1]-σ)/2)*
                gamma(1+(θμ[2]-θμ[1]+σ)/2)*rgamma(1+(θμ[2]-θμ[1]-σ)/2);
    return sqrt(-ess*(Γfac))
end

function Invseries(ser) #ser only being a 2x2 matrix vector
    Inverse = zeros(CC, 2, 2, N+1);
    Inverse[:,:,1] = [ CC(1) CC(0) ; CC(0) CC(1) ];
    for i = 2:(N+1)
        A = -ser[:,:,i];
        for j = 2:(i-1)
            A += - (ser[:,:,j]*Inverse[:,:,(1+i-j)]);
        end
        Inverse[:,:,i] = A
    end
    return Inverse
end

function Gee(σ,θ1,θ2)
    psi = zeros(CC,2,2, N+1);
    a = (-σ-θ1-θ2)/2;
    b = (-σ-θ1+θ2)/2;
    c = -σ
    psi[:,:,1] = [ CC(1) CC(0) ; CC(0) CC(1) ];
    psi[:,:,2] = [ ((a*b)/c) (-(a-c)*(b-c)/(c*(1-c))) ; (a*b/((1+c)*c)) (-(a-c)*(b-c)/c) ];
    for p = 3:(N+1)
        # this transposes assignment for some reason
        psi[:,:,p] = ([ ((a+p-2)*(b+p-2)/((p-1)*(c+p-2)))*psi[1,1,p-1]
                      ((a+p-2)*(b+p-2)/((p-2)*(c+p-1)))*psi[2,1,p-1] ;
                      ((a-c+p-2)*(b-c+p-2)/((p-1-c)*(p-2)))*psi[1,2,p-1]
                      ((a-c+p-2)*(b-c+p-2)/((p-1)*(-c+p-2)))*psi[2,2,p-1] ])
    end
    return psi
end

function GeeV(σ,θ1)
    psi = zeros(CC,2,2, N+1);
    a = (σ-θ1)/2;
    psi[:,:,1] = [ CC(1) CC(0) ; CC(0) CC(1) ];
    psi[:,:,2] = [ (a/σ) -a/(σ*(σ-1)) ; (σ-a)/((1+σ)*σ) -(a-σ)/σ ];
    for p = 3:(N+1)
        # this transposes assignment for some reason
        psi[:,:,p] = ([ ((a+p-2)/((p-1)*(σ+p-2)))*psi[1,1,p-1]
                      ((a+p-2)/((p-2)*(σ+p-1)))*psi[2,1,p-1] ;
                      ((a-σ+p-2)/((p-1-σ)*(p-2)))*psi[1,2,p-1]
                      ((a-σ+p-2)/((p-1)*(-σ+p-2)))*psi[2,2,p-1] ])
    end
    return psi
end

function Aparallel(ind, vecg, vecginv)
    result = zeros(CC,2,2);
    if ind[2] > (ind[1]-1)
         Threads.@threads for i = 1:(ind[1])
            result += -vecginv[:,:,(N-ind[2]+i+1)]*vecg[:,:,(ind[1]-i+1)];
        end
    end
    return result
end

function BuildA(θμ, σ)
    vecg = GeeV(σ, θμ[3])
    vecginv = Invseries(vecg);
    ind = [(p,q) for p in range(1,length=N) for q in range(1,length=N)];
    OPA = zeros(CC,2,2, N^2);
    Threads.@threads for i = 1:(N^2)
    #for i = 1:(N^2)
        OPA[:,:,i] = Aparallel(ind[i], vecg, vecginv)
    end
    A = zeros(CC,2N,2N);
    Threads.@threads for p = 1:N
        Threads.@threads for q = 1:N
            A[(2p-1):(2p), (2q-1):(2q)] = OPA[:,:,N*(p-1)+q];
        end
    end
    return A
end

function Dparallel(ind, vecg, vecginv)
    result = zeros(CC,2,2);
    if ind[2] < (ind[1]+1)
        Threads.@threads for i = 1:(N-ind[1]+1)
            result += -vecginv[:,:,(ind[2]+i)]*vecg[:,:,(N-ind[1]-i+2)];
        end
    end
    return result
end

function BuildD(θμ, σ)
    vecg = Gee( σ, θμ[2], θμ[1]);
    vecginv = Invseries(vecg);
    ind = [(p,q) for p in range(1,length=N) for q in range(1,length=N)];
    OPD = zeros(CC,2,2, N^2)
    Threads.@threads for i = 1:(N^2)
        OPD[:,:,i] = Dparallel(ind[i], vecg, vecginv)
    end
    D = zeros(CC,2N,2N);
    Threads.@threads for p = 1:N
        Threads.@threads for q = 1:N
            D[(2p-1):(2p), (2q-1):(2q)] = OPD[:, :, N*(p-1)+q];
        end
    end
    return D
end

function MultD(OpD, σ, η, t)
    tm = [ CC((t^(σ/2))*(η)) CC(0) ; CC(0) CC((t^(-σ/2))/(η)) ];
    tminv = [ CC((t^(-σ/2))/(η)) CC(0) ; CC(0) CC((t^(σ/2))*(η)) ];
    MultD = zeros(CC, 2N, 2N);
    Threads.@threads for p = 1:N
        Threads.@threads for q = 1:N
            tid = [ CC(t^(N+q-p)) CC(0) ; CC(0) CC((t^(N+q-p))) ];
            MultD[(2p-1):(2p), (2q-1):(2q)] = (tid*(tm*(OpD[(2p-1):(2p), (2q-1):(2q)])*tminv));
        end
    end
    return MultD
end

function MultDdiff(OpD, σ, η, t)
    tm = [ CC((t^(σ/2))*(η)) CC(0) ; CC(0) CC((t^(-σ/2))/(η)) ];
    tminv = [ CC((t^(-σ/2))/(η)) CC(0) ; CC(0) CC((t^(σ/2))*(η)) ];
    sigma3 = [ CC(1) CC(0) ; CC(0) CC(-1) ]
    result = zeros(CC, 2N, 2N)
    Threads.@threads for p = 1:N
        Threads.@threads for q = 1:N
            tid = [ CC(t^(N-p+q)) CC(0) ; CC(0) CC(t^(N-p+q)) ];
            X = tid*tm*OpD[(2p-1):(2p),(2q-1):(2q)]*tminv;
            A = [ CC((N-p+q)/t) CC(0) ; CC(0) CC((N-p+q)/t) ]*X
            B = [ CC(σ/(2*t)) CC(0) ; CC(0) CC(σ/(2*t)) ]*(sigma3*X-X*sigma3)
            # it seems Julia gets confused with operation hierarchy...
            result[(2p-1):(2p),(2q-1):(2q)] = A + B
        end
    end
    return result
end

function MultDdiff2(OpD, σ, η, t) # not fully tested
    tm = S([ CC((t^(σ/2))*(η)) CC(0) ; CC(0) CC((t^(-σ/2))/(η)) ]);
    tminv = S([ CC((t^(σ/2))/(η)) CC(0) ; CC(0) CC((t^(σ/2))*(η)) ]);
    sigma3 = S([ CC(1) CC(0) ; CC(0) CC(-1) ]);
    result = zeros(CC, 2N, 2N)
    Threads.@threads for p = 1:N
        Threads.@threads for q = 1:N
            tid = S([ CC(t^(N-p+q)) CC(0) ; CC(0) CC((t^(N-p+q))) ]);
            X = tid*tm*S(OpD[(2p-1):(2p),(2q-1):(2q)])*tminv
            A = S([ CC((N-p+q)*(N-p+q-1.0)/t^2) CC(0) ; CC(0) CC((N-p+q)*(N-p+q-1.0)/t^2) ])*X
            B = S([ CC(σ^2/(2*t^2)) CC(0) ; CC(0) CC(σ^2/(2*t^2)) ])*(X-sigma3*X*sigma3)
            C = S([ CC(σ/(2*t^2)-(N-p+q)*σ/(t^2)) CC(0) ; CC(0) CC(σ/(2*t^2)-(N-p+q)*σ/(t^2)) ])*(sigma3*X-X*sigma3)
            result[(2p-1):(2p),(2q-1):(2q)] = A + B + C
        end
    end
    return result
end

function firstlogdiffV(OpA,OpD,σ, θμ, η, t)
    OPK = S(MultD(OPD,σ,η,t))*S(OpA)
    OPKdiff = S(MultDdiff(OPD,σ,η,t))*S(OpA)
    BIGM = -inv(one(S)-OPK)*OPKdiff
    value = tr(BIGM)
    return (σ^2-θμ[1]^2-θμ[2]^2)/(4*t)-θμ[2]/2+value
end

function secondlogdiffVI(OpA,OpD,σ, θμ, η, t) # not fully tested
    OPK = S(OpA)*S(MultD(OPD,σ[1],η,t))
    OPKdiff = S(OpA)*S(MultDdiff(OPD,σ[1],η,t))
    OPKdiff2 = S(OpA)*S(MultDdiff2(OPD,σ[1],η,t))
    Minv = -inv(one(S)-OPK)
    value1 = tr(Minv*OPKdiff)
    value2 = tr(Minv*OPKdiff*Minv*OPKdiff+Minv*OPKdiff2)
    value = -(2t-1)*value1-t*(t-1)*value2
    return (σ^2-(θμ[1]+θμ[2])^2)/4-θμ[2]/2+value
end

function τV(OpA,OpD, σ, θμ, η, t)
    MD = MultD(OpD, σ, η, t);
    OPK = (S(MD)*S(OpA));
    Fred = (one(S)-OPK);
    result = (t^(((σ^2) - (θμ[1]^2) - (θμ[2]^2))/4))*(exp(-θμ[2]/2*t))*(det(Fred));
    return result
end

# this computes the accessory parameter as defined in paper
function accessoryV(OpA,OpD, σ, θμ, η, t)
    prefactor = -θμ[1]*θμ[2]/(2*t);
    return firstlogdiffV(OpA,OpD, σ, θμ, η, t)+prefactor
end

# test values from GIL
t = CC(1.0)
θs = [ CC(0.3804+0.6212*onei(CC)) CC(0.8364-0.4218*onei(CC)) CC(0.6858+0.6628*onei(CC)) CC(0.0326+0.3610*onei(CC)) ]
σs = [ CC(-0.6544-0.9622*onei(CC)) CC(0.1916+0.6336*onei(CC)) CC(0.9524+0.2056*onei(CC)) ]

η = computeη(θs, σs[1], σs[2])
OPA = BuildA(θs, σs[1])
OPD = BuildD(θs, σs[1])
#Painleveτ = τV(OPA,OPD, σs[1], θs, η, t)
#println(Painleveτ)
key = accessoryV(OPA,OPD, σs[1], θs, η, t)
println(key)
