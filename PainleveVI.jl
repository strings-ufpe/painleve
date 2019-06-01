using LinearAlgebra
using Base.Threads
using Nemo

# This implments the Fredholm determinant expansion for the Painlevé VI
# tau function, numerically. It is written in arbitrary precision routine

JULIA_NUM_THREADS=4
N = 64
CC = ComplexField(256)
S = MatrixSpace(CC,2N,2N)

function frickejimbo(Pμ, Pμν)
    b = Pμν[1]*Pμν[2] - Pμ[1]*Pμ[3] - Pμ[2]*Pμ[4];
    c = (Pμν[1])^2 + (Pμν[2])^2 + (Pμ[1])^2 + (Pμ[2])^2 + (Pμ[3])^2 +
        (Pμ[4])^2 +(Pμ[1]*Pμ[2]*Pμ[3]*Pμ[4])-Pμν[1]*(Pμ[1]*Pμ[2]+Pμ[3]*Pμ[4]) -
        Pμν[2]*(Pμ[1]*Pμ[4]+Pμ[2]*Pμ[3])-4.0;

     return (-b - sqrt((b^2 - 4c)))/2
end

function computeη(θμ, θμν, t0 = 0)
    Pμ = [ 2*(cospi(x)) for x in θμ ]
    Pμν = [ 2*(cospi(x)) for x in θμν ]

    Pμν[3] = frickejimbo(Pμ, Pμν)

    if (t0 == 0)
        prefactor = Pμ[2]*Pμ[3] + Pμ[1]*Pμ[4] - 2Pμν[2] - Pμν[1]*Pμν[3] -
                    (Pμ[1]*Pμ[3] + Pμ[2]*Pμ[4] - 2Pμν[3] - Pμν[1]*Pμν[2])*
                    (exppii(θμν[1]));
    else
        prefactor = Pμ[2]*Pμ[3] + Pμ[1]*Pμ[4] - 2Pμν[2] - Pμν[3]*Pμν[1] -
                    (Pμ[1]*Pμ[3] + Pμ[2]*Pμ[4] - 2Pμν[3] - Pμν[2]*Pμν[1])*
                    (exppii(-θμν[1]));
    end

    hugeΓfac = (gamma(1-θμν[1])^2)*gamma(1+(θμ[2]+θμ[1]+θμν[1])/2)*
                gamma(1+(θμ[2]-θμ[1]+θμν[1])/2)*gamma(1+(θμ[3]+θμ[4]+θμν[1])/2)*
                gamma(1+(θμ[3]-θμ[4]+θμν[1])/2)/
                (16*(pi^2)*sinpi((θμν[1]+θμ[1]-θμ[2])/2)*
                sinpi((θμν[1]-θμ[3]+θμ[4])/2))*
                gamma((θμν[1]-θμ[1]-θμ[2])/2)*gamma((θμν[1]-θμ[3]-θμ[4])/2)*
                (rgamma(1+θμν[1])^2)*rgamma(1+(θμ[2]-θμ[1]-θμν[1])/2)*
                rgamma(1+(θμ[3]-θμ[4]-θμν[1])/2);
     #gamma(θμν[1]- θμ[1] - θμ[2])
    return (hugeΓfac*prefactor)
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

function Gee(σ,θ1, θ2)
    psi = zeros(CC,2,2, N+1);
    a = (σ-θ1+θ2)/2;
    b = (σ-θ1-θ2)/2;
    psi[:,:,1] = [ CC(1) CC(0) ; CC(0) CC(1) ];
    psi[:,:,2] = [ ((a*b)/σ) (((-a)*b)/((1+σ)*σ)) ; (((σ-a)*(σ-b))/((1-σ)*σ)) (((σ-a)*(b-σ))/σ) ];
    for p = 3:(N+1)
        # this transposes assignment for some reason
        psi[:,:,p] = ([ ((a+p-2)*(b+p-2)/((p-1)*(σ+p-2)))*psi[1,1, p-1]
                      ((a-σ+p-2)*(b-σ+p-2)/((p-1-σ)*(p-2)))*psi[2,1, p-1] ;
                      ((a+p-2)*(b+p-2)/((p-2)*(σ+p-1)))*psi[1,2, p-1]
                      ((a-σ+p-2)*(b-σ+p-2)/((p-1)*(p-σ-2)))*psi[2,2, p-1] ])
    end
    return psi
end

function Aparallel(ind, vecg, vecginv)
    result = zeros(CC,2,2);
    if ind[2] < (ind[1]+1)
         Threads.@threads for i = 1:(N-ind[1]+1)
            result += -vecg[:,:,(N+2-ind[1]-i)]*vecginv[:,:,(ind[2]+i)];
        end
    end
    return result
end

function BuildA(θμ, σ)
    vecg = Gee(σ[1], θμ[3], θμ[4])
    vecginv = Invseries(vecg);
    ind = [(p,q) for p in range(1,length=N) for q in range(1,length=N)];
    OPA = zeros(CC,2,2, N^2);
    Threads.@threads for i = 1:(N^2)
    #for i = 1:(N^2)
        OPA[:,:,i] = Aparallel(ind[i], vecg, vecginv)
    end
    A = zeros(CC,2N,2N);
    Threads.@threads for p = 1:N
    #for p = 1:N
        Threads.@threads for q = 1:N
        #for q = 1:N
            #result = zeros(CC,2,2)
            #if q < (p+1)
            #    Threads.@threads for i = 1:(N-p+1)
            #        result += -vecg[:,:,(N+2-p-i)]*vecginv[:,:,(q+i)];
            #    end
            #end
            #A[ (2p-1):2p, (2q-1):(2q) ] = result
            A[(2p-1):(2p), (2q-1):(2q)] = OPA[:,:,N*(p-1)+q];
        end
    end
    return A
end

function Dparallel(ind, vecg, vecginv)
    result = zeros(CC,2,2);
    if ind[2] > (ind[1]-1)
        Threads.@threads for i = 1:ind[1]
            result += -vecg[:,:,(ind[1]-i+1)]*vecginv[:,:,(N - ind[2]+i+1)];
        end
    end
    return result
end

function BuildD(θμ, σ)
    vecg = Gee( ((-1)*(σ[1])), θμ[2], θμ[1]);
    vecginv = Invseries(vecg);
    ind = [(p,q) for p in range(1,length=N) for q in range(1,length=N)];
    OPD = zeros(CC,2,2, N^2)
    Threads.@threads for i = 1:(N^2)
    #for i = 1:(N^2)
        OPD[:,:,i] = Dparallel(ind[i], vecg, vecginv)
    end
    D = zeros(CC,2N,2N);
    Threads.@threads for p = 1:N
    #for p = 1:N
        Threads.@threads for q = 1:N
        #for q = 1:N
            D[(2p-1):(2p), (2q-1):(2q)] = OPD[:, :, N*(p-1)+q];
        end
    end
    return D
end

function MultD(OpD, σ, η, t)
    tm = [ CC((t^(-σ/2))/(sqrt(η))) CC(0) ; CC(0) CC((t^(σ/2))*(sqrt(η))) ];
    tminv = [ CC((t^(σ/2))*(sqrt(η))) CC(0) ; CC(0) CC((t^(-σ/2))/(sqrt(η))) ];
    MultD = zeros(CC, 2N, 2N);
    Threads.@threads for p = 1:N
    #for p = 1:N
        Threads.@threads for q = 1:N
        #for q = 1:N
            tid = [ CC(t^(N+p-q)) CC(0) ; CC(0) CC((t^(N+p-q))) ];
            MultD[(2p-1):(2p), (2q-1):(2q)] = (tid*(tm*(OpD[(2p-1):(2p), (2q-1):(2q)])*tminv));
        end
    end
    return MultD
end

function τVI(OpA,OpD, σ, θμ, η, t)
    MD = MultD(OpD, σ[1], η, t);
    OPK = (S(OpA)*S(MD));
    Fred = (one(S)-OPK);
    τ = (t^(((σ[1]^2) - (θμ[1]^2) - (θμ[2]^2))/4))*((1-t)^(-θμ[2]*θμ[3]/2))*(det(Fred));
    return τ
end

function MultDdiff(OpD, σ, η, t)
    tm = [ CC((t^(-σ/2))/(sqrt(η))) CC(0) ; CC(0) CC((t^(σ/2))*(sqrt(η))) ];
    tminv = [ CC((t^(σ/2))*(sqrt(η))) CC(0) ; CC(0) CC((t^(-σ/2))/(sqrt(η))) ];
    sigma3 = [ CC(1) CC(0) ; CC(0) CC(-1) ]
    result = zeros(CC, 2N, 2N)
    Threads.@threads for p = 1:N
        Threads.@threads for q = 1:N
            tid = [ CC(t^(N+p-q)) CC(0) ; CC(0) CC(t^(N+p-q)) ];
            X = tid*tm*OpD[(2p-1):(2p),(2q-1):(2q)]*tminv;
            A = [ CC((N+p-q)/t) CC(0) ; CC(0) CC((N+p-q)/t) ]*X
            B = [ CC(σ/(2*t)) CC(0) ; CC(0) CC(σ/(2*t)) ]*(sigma3*X-X*sigma3)
            # it seems Julia gets confused with operation hierarchy...
            result[(2p-1):(2p),(2q-1):(2q)] = A - B
        end
    end
    return result
end

function MultDdiff2(OpD, σ, η, t)
    tm = [ CC((t^(-σ/2))/(sqrt(η))) CC(0) ; CC(0) CC((t^(σ/2))*(sqrt(η))) ];
    tminv = [ CC((t^(σ/2))*(sqrt(η))) CC(0) ; CC(0) CC((t^(-σ/2))/(sqrt(η))) ];
    sigma3 = [ CC(1) CC(0) ; CC(0) CC(-1) ];
    result = zeros(CC, 2N, 2N)
    Threads.@threads for p = 1:N
        Threads.@threads for q = 1:N
            tid = [ CC(t^(N+p-q)) CC(0) ; CC(0) CC((t^(N+p-q))) ];
            X = tid*tm*OpD[(2p-1):(2p),(2q-1):(2q)]*tminv
            A = [ CC((N+p-q)*(N+p-q-1.0)/t^2) CC(0) ; CC(0) CC((N+p-q)*(N+p-q-1.0)/t^2) ]*X
            B = [ CC(σ^2/(2*t^2)) CC(0) ; CC(0) CC(σ^2/(2*t^2)) ]*(X-sigma3*X*sigma3)
            C = [ CC(σ/(2*t^2)-(N+p-q)*σ/(t^2)) CC(0) ; CC(0) CC(σ/(2*t^2)-(N+p-q)*σ/(t^2)) ]*(sigma3*X-X*sigma3)
            result[(2p-1):(2p),(2q-1):(2q)] = A + B + C
        end
    end
    return result
end

function firstlogdiffVI(OpA,OpD,σ, θμ, η, t)
    OPK = S(OpA)*S(MultD(OPD,σ[1],η,t))
    OPKdiff = S(OpA)*S(MultDdiff(OPD,σ[1],η,t))
    BIGM = -inv(one(S)-OPK)*OPKdiff
    value = tr(BIGM)
    return (σ[1]^2-θμ[1]^2-θμ[2]^2)/(4*t)-θμ[2]*θμ[3]/(2*(t-1))+value
end

function secondlogdiffVI(OpA,OpD,σ, θμ, η, t)
    OPK = S(OpA)*S(MultD(OPD,σ[1],η,t))
    OPKdiff = S(OpA)*S(MultDdiff(OPD,σ[1],η,t))
    OPKdiff2 = S(OpA)*S(MultDdiff2(OPD,σ[1],η,t))
    Minv = -inv(one(S)-OPK)
    value1 = tr(Minv*OPKdiff)
    value2 = tr(Minv*OPKdiff*Minv*OPKdiff+Minv*OPKdiff2)
    value = -(2t-1)*value1-t*(t-1)*value2
    return (σ[1]^2-θμ[1]^2-θμ[2]^2)/4-θμ[2]*θμ[3]/2+value
end

function σVI(OpA,OpD, σ, θμ, η, t)
    prefactor = -(θμ[1]^2+θμ[2]^2-θμ[3]^2-θμ[4]^2)/8+(θμ[2]^2-θμ[4]^2)/4*t;
    return t*(t-1)*firstlogdiffVI(OpA,OpD, σ, θμ, η, t)+prefactor
end

#t = CC(0.4);
#θs = [ CC(1.0/(2.0*sqrt(2.0))) CC(1.0/(2.0*sqrt(2.0))*onei(CC)) CC(1.0/(2.0*sqrt(2.0))) CC(-2.0) ]
#σs = [ CC(1/(2sqrt(2))*(2+onei(CC))) CC((1/(2sqrt(2)))*(2+onei(CC))) CC(1) ]
t = CC(0.1)
θs = [ CC(0.3804+0.6212*onei(CC)) CC(0.8364-0.4218*onei(CC)) CC(0.6858+0.6628*onei(CC)) CC(0.0326+0.3610*onei(CC)) ]
σs = [ CC(-0.6544-0.9622*onei(CC)) CC(0.1916+0.6336*onei(CC)) CC(0.9524+0.2056*onei(CC)) ]

η = computeη(θs, σs)
OPA = BuildA(θs, σs)
OPD = BuildD(θs, σs)
#println(η)
#Painleveτ = τVI(OPA,OPD, σs, θs, η, t)
#println(Painleveτ)
Painσ = σVI(OPA,OPD, σs, θs, η, t)
println(Painσ)
