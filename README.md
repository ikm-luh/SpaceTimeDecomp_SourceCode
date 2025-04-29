# SpaceTimeDecomp_SourceCode
Corresponding Code to the paper "A fully separated space-time decomposition for viscoelasticity" by Hendrik Geisler, David Néron and Philipp Junker.
For all boundary value examples, the code for the PGD computation and the usual time-step-wise Newton-Raphson method are presented.

The code implementation for the femur example is the most thoroughly documented. 

For the FEM computation, the Julia package "Ferrite.jl" is used.
The input file reader src/inp_reader has been developed by Dustin Jantos.

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> SpaceTimeDecomp_SourceCode

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "SpaceTimeDecomp_SourceCode"
```
which auto-activate the project and enable local path handling from DrWatson.
