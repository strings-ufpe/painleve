module Painleve

greet() = print("This package implements arbitrary-precision expansions (near t=0) for the Painlevé V and VI transcendents")

export pieV,
       pieVI,
       ess,
       tauhatVI,
       tauVI,
       accessoryKVI,
       hirotaVI,
       tauhatV,
       tauV,
       accessoryKV,
       hirotaV

import ArbNumerics
import LinearAlgebra

include("Fredholm.jl")

end # module
