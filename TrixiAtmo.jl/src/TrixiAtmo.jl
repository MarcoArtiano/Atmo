"""
    🌍 TrixiAtmo 🌍

**TrixiAtmo.jl** is a simulation package for atmospheric models based on
[Trixi.jl](https://github.com/trixi-framework/Trixi.jl)

See also: [trixi-framework/TrixiAtmo.jl](https://github.com/trixi-framework/TrixiAtmo.jl)
"""
module TrixiAtmo

using Trixi
using MuladdMacro: @muladd
using Static: True, False
using StrideArrays: PtrArray
using StaticArrayInterface: static_size
using LinearAlgebra: norm
using Reexport: @reexport
@reexport using StaticArrays: SVector

include("auxiliary/auxiliary.jl")
include("equations/equations.jl")
include("meshes/meshes.jl")
include("solvers/solvers.jl")
include("semidiscretization/semidiscretization_hyperbolic_2d_manifold_in_3d.jl")

export CompressibleMoistEulerEquations2D
export CompressibleEulerEquations1DNC
export CompressibleEulerPotentialTemperatureEquations1D
export CompressibleEulerPotentialTemperatureEquations2D
export CompressibleEulerPotentialTemperatureEquations3D
export CompressibleEulerPotentialTemperatureEquations1DNC
export CompressibleEulerPotentialTemperatureEquations2DNC
export LinearizedGravityWaveEquations2D
export LinearizedGravityWaveEquationsSlow2D
export LinearizedGravityWaveEquationsFast2D

export flux_chandrashekar, flux_theta, flux_theta_rhoAM, flux_theta_physentropy,
       flux_theta_physentropy2, flux_theta_es

export flux_theta_global, flux_theta_AM, flux_theta_rhos, flux_theta_rhos_AM, flux_theta_global_es, flux_theta_rhos_es, flux_LMARS, flux_lmars2_no_advection
export flux_theta_global_es, flux_theta_AM_es, flux_theta_rhos_es, flux_theta_rhos_AM_es
export flux_nonconservative_gravity, flux_lmars

export examples_dir

end # module TrixiAtmo
