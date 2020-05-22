# Painlevé

These incorporate implementations of the expansion of isomonodromic tau functions in Julia
Python is included for legacy.

The goal is to implement arbitrary-precision evaluation for the Painlevé tau functions.
This project uses ArbNumerics (https://github.com/JeffreySarnoff/ArbNumerics.jl)

The method uses the Fredholm determinant formulation for the Painlevé VI and V given in
Comm. Math. Phys. 363 (2018) 1-58 and J. Math. Phys. 59 (2018) 091409
truncated to Nf=48 Fourier components at small isomonodromic parameters.

Long term goals:
- Transparent calculation of Painlevé III, V and VI tau function for any value of the parameters
- Implementation of isomonodromic tau function for a generic number of regular singular points

Done (some scripts are in the legacy folder):
- Small parameter expansion in terms of Nekrasov functions for Painlevé V and VI (slow, reference)
- Small parameter expansion in terms of Fredholm determinants for Painlevé V and VI (fast)
- Small parameter expansion of Painlevé V and VI in SciPy. Includes accessory parameter expansion

Immediate goals:
- Large parameter expansion for Painlevé V
- Five regular singular points, with asymptotic expansion for the accessory parameter.
