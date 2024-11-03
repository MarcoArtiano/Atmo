using Trixi: AbstractEquations

abstract type AbstractCompressibleMoistEulerEquations{NDIMS, NVARS} <:
              AbstractEquations{NDIMS, NVARS} end
abstract type AbstractCompressibleEulerPotentialTemperatureEquations{NDIMS, NVARS} <:
              AbstractEquations{NDIMS, NVARS} end
abstract type AbstractLinearizedEulerEquations{NDIMS, NVARS} <: 
              AbstractEquations{NDIMS,NVARS} end
include("compressible_moist_euler_2d_lucas.jl")
include("compressible_euler_potential_temperature_1d.jl")
include("compressible_euler_potential_temperature_2d.jl")
include("compressible_euler_potential_temperature_3d.jl")
include("compressible_euler_potential_temperature_1d_nc.jl")
include("compressible_euler_potential_temperature_2d_nc.jl")
include("compressible_euler_1d_nc.jl")


                    include("linear_gravity_wave_2d.jl")
                    include("linear_boussinesque_fast_2d.jl")
                    include("linear_boussinesque_slow_2d.jl")
                    include("linear_boussinesque_fast_vertical_2d.jl")