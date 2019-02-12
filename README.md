# Painlevé

These incorporate implementations of the expansion of isomonodromic tau functions in python/Julia

Long term goals: 
- Transparent calculation of Painlevé V and VI tau function for any value of the parameters
- Implementation of isomonodromic tau function for a generic number of regular singular points

Done:
- Small parameter expansion in terms of Nekrasov functions for Painlevé V and VI (slow, reference)
- Small parameter expansion in terms of Fredholm determinants for Painlevé VI (fast)
- Small parameter expansion of Painlevé V and VI in SciPy. Includes accessory parameter expansion

Immediate goals:
- Fredholm determinant expansion for Painlevé V
- Faster numerics (Julia vs Python)
- Large parameter expansion for Painlevé V
- Five regular singular points, with asymptotic expansion for the accessory parameter.
