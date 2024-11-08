using Trixi
using Trixi: ln_mean, stolarsky_mean
import Trixi: varnames, cons2cons, cons2prim, cons2entropy, entropy, FluxLMARS

@muladd begin
#! format: noindent
struct CompressibleEulerPotentialTemperatureEquations3D{RealT <: Real} <:
       AbstractCompressibleEulerPotentialTemperatureEquations{3, 5}
    p_0::RealT
    c_p::RealT
    c_v::RealT
    g::RealT
    R::RealT
    gamma::RealT
    inv_gamma_minus_one::RealT
    K::RealT
end

function CompressibleEulerPotentialTemperatureEquations3D(; g = 9.81, RealT = Float64)
    p_0 = 100_000.0
    c_p = 1004.0
    c_v = 717.0
    R = c_p - c_v
    gamma = c_p / c_v
    inv_gamma_minus_one = inv(gamma-1)
    K = p_0 * (R / p_0)^gamma

    return CompressibleEulerPotentialTemperatureEquations3D{RealT}(p_0, c_p, c_v, g, R,
                                                                   gamma, inv_gamma_minus_one, K)
end

function varnames(::typeof(cons2cons),
                  ::CompressibleEulerPotentialTemperatureEquations3D)
    ("rho", "rho_v1", "rho_v2", "rho_v3", "rho_theta")
end

varnames(::typeof(cons2prim), ::CompressibleEulerPotentialTemperatureEquations3D) = ("rho",
                                                                                     "v1",
                                                                                     "v2",
                                                                                     "v3",
                                                                                     "p1")

# Calculate 1D flux for a single point in the normal direction.
# Note, this directional vector is not normalized.
@inline function flux(u, normal_direction::AbstractVector,
                      equations::CompressibleEulerPotentialTemperatureEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_theta = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2] + v3* normal_direction[3]
    rho_v_normal = rho * v_normal
    f1 = rho_v_normal
    f2 = rho_v_normal * v1 + p * normal_direction[1]
    f3 = rho_v_normal * v2 + p * normal_direction[2]
    f4 = rho_v_normal * v3 + p * normal_direction[3]
    f5 = rho_theta * v_normal
    return SVector(f1, f2, f3, f4, f5)
end

@inline function flux(u, orientation::Integer,
                      equations::CompressibleEulerPotentialTemperatureEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_theta = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    p = equations.K * rho_theta^equations.gamma
    if orientation == 1
        f1 = rho_v1
        f2 = rho_v1 * v1 + p
        f3 = rho_v1 * v2
        f4 = rho_v1 * v3
        f5 = rho_theta * v1
    elseif orientation == 2
        f1 = rho_v2
        f2 = rho_v2 * v1
        f3 = rho_v2 * v2 + p
        f4 = rho_v2 * v3
        f5 = rho_theta * v2
    else
        f1 = rho_v3
        f2 = rho_v3 * v1
        f3 = rho_v3 * v2 
        f4 = rho_v3 * v3 + p
        f5 = rho_theta * v3
    end

    return SVector(f1, f2, f3, f4, f5)
end

@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
    x, t,
    surface_flux_function,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
norm_ = norm(normal_direction)
# Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
normal = normal_direction / norm_

# Some vector that can't be identical to normal_vector (unless normal_vector == 0)
tangent1 = SVector(normal_direction[2], normal_direction[3], -normal_direction[1])
# Orthogonal projection
tangent1 -= dot(normal, tangent1) * normal
tangent1 = normalize(tangent1)

# Third orthogonal vector
tangent2 = normalize(cross(normal_direction, tangent1))

# rotate the internal solution state
u_local = rotate_to_x(u_inner, normal, tangent1, tangent2, equations)

# compute the primitive variables
rho_local, v_normal, v_tangent1, v_tangent2, p_local = cons2prim(u_local, equations)

# Get the solution of the pressure Riemann problem
# See Section 6.3.3 of
# Eleuterio F. Toro (2009)
# Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
# [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
if v_normal <= 0.0
sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
p_star = p_local *
(1 + 0.5 * (equations.gamma - 1) * v_normal / sound_speed)^(2 *
                                   equations.gamma *
                                   equations.inv_gamma_minus_one)
else # v_normal > 0.0
A = 2 / ((equations.gamma + 1) * rho_local)
B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
p_star = p_local +
0.5 * v_normal / A *
(v_normal + sqrt(v_normal^2 + 4 * A * (p_local + B)))
end

# For the slip wall we directly set the flux as the normal velocity is zero
return SVector(zero(eltype(u_inner)),
p_star * normal[1],
p_star * normal[2],
p_star * normal[3],
zero(eltype(u_inner))) * norm_
end

"""
boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
surface_flux_function, equations::CompressibleEulerEquations3D)

Should be used together with [`TreeMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, orientation,
    direction, x, t,
    surface_flux_function,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
# get the appropriate normal vector from the orientation
if orientation == 1
normal_direction = SVector(1.0, 0.0, 0.0)
elseif orientation == 2
normal_direction = SVector(0.0, 1.0, 0.0)
else # orientation == 3
normal_direction = SVector(0.0, 0.0, 1.0)
end

# compute and return the flux using `boundary_condition_slip_wall` routine above
return boundary_condition_slip_wall(u_inner, normal_direction, direction,
x, t, surface_flux_function, equations)
end

"""
boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t,
surface_flux_function, equations::CompressibleEulerEquations3D)

Should be used together with [`StructuredMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
    direction, x, t,
    surface_flux_function,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
# flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
# to be inward pointing on the -x, -y, and -z sides due to the orientation convention used by StructuredMesh
if isodd(direction)
boundary_flux = -boundary_condition_slip_wall(u_inner, -normal_direction,
            x, t, surface_flux_function,
            equations)
else
boundary_flux = boundary_condition_slip_wall(u_inner, normal_direction,
           x, t, surface_flux_function,
           equations)
end

return boundary_flux
end

@inline function source_terms_gravity(u, x, t,
                                      equations::CompressibleEulerPotentialTemperatureEquations3D)
    rho, _, _, _, _= u
    return SVector(zero(eltype(u)), zero(eltype(u)),
                   zero(eltype(u)), -equations.g * rho, zero(eltype(u)))
end

@inline function rotate_to_x(u, normal_vector, tangent1, tangent2,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
# Multiply with [ 1   0        0       0   0;
#                 0   ―  normal_vector ―   0;
#                 0   ―    tangent1    ―   0;
#                 0   ―    tangent2    ―   0;
#                 0   0        0       0   1 ]
return SVector(u[1],
normal_vector[1] * u[2] + normal_vector[2] * u[3] +
normal_vector[3] * u[4],
tangent1[1] * u[2] + tangent1[2] * u[3] + tangent1[3] * u[4],
tangent2[1] * u[2] + tangent2[2] * u[3] + tangent2[3] * u[4],
u[5])
end

# Low Mach number approximate Riemann solver (LMARS) from
# X. Chen, N. Andronova, B. Van Leer, J. E. Penner, J. P. Boyd, C. Jablonowski, S.
# Lin, A Control-Volume Model of the Compressible Euler Equations with a Vertical Lagrangian
# Coordinate Monthly Weather Review Vol. 141.7, pages 2526–2544, 2013,
# https://journals.ametsoc.org/view/journals/mwre/141/7/mwr-d-12-00129.1.xml.

@inline function (flux_lmars::FluxLMARS)(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::CompressibleEulerPotentialTemperatureEquations3D)
    a = flux_lmars.speed_of_sound
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)

    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] + v3_ll * normal_direction[3]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] + v3_rr * normal_direction[3]

    norm_ = norm(normal_direction)

    rho = 0.5f0 * (rho_ll + rho_rr)

    p_interface = 0.5f0 * (p_ll + p_rr) - 0.5f0 * a * rho * (v_rr - v_ll) / norm_
    v_interface = 0.5f0 * (v_ll + v_rr) - 1 / (2 * a * rho) * (p_rr - p_ll) * norm_

    if (v_interface > 0)
        f1, f2, f3, f4, f5 = u_ll * v_interface
    else
        f1, f2, f3, f4, f5 = u_rr * v_interface
    end

    return SVector(f1,
                   f2 + p_interface * normal_direction[1],
                   f3 + p_interface * normal_direction[2],
                   f4 + p_interface * normal_direction[3],
                   f5)
end

@inline function (flux_lmars::FluxLMARS)(u_ll, u_rr, orientation::Integer,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
c = flux_lmars.speed_of_sound

# Unpack left and right state
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)

if orientation == 1
v_ll = v1_ll
v_rr = v1_rr
elseif orientation == 2
v_ll = v2_ll
v_rr = v2_rr
else # orientation == 3
v_ll = v3_ll
v_rr = v3_rr
end

rho = 0.5 * (rho_ll + rho_rr)
p = 0.5 * (p_ll + p_rr) - 0.5 * c * rho * (v_rr - v_ll)
v = 0.5 * (v_ll + v_rr) - 1 / (2 * c * rho) * (p_rr - p_ll)

# We treat the energy term analogous to the potential temperature term in the paper by
# Chen et al., i.e. we use p_ll and p_rr, and not p
if v >= 0
f1, f2, f3, f4, f5 = v * u_ll
else
f1, f2, f3, f4, f5 = v * u_rr
end

if orientation == 1
f2 += p
elseif orientation == 2
f3 += p
else # orientation == 3
f4 += p
end

return SVector(f1, f2, f3, f4, f5)
end

## Entropy (total energy) conservative flux for the Compressible Euler with the Potential Formulation

@inline function flux_theta(u_ll, u_rr, normal_direction::AbstractVector,
                            equations::CompressibleEulerPotentialTemperatureEquations3D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] + v3_ll * normal_direction[3]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] + v3_rr * normal_direction[3]
    _, _, _, _, rho_theta_ll = u_ll
    _, _, _, _, rho_theta_rr = u_rr
    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)

    gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v3_avg = 0.5f0 * (v3_ll + v3_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    f2 = f1 * v1_avg + p_avg * normal_direction[1]
    f3 = f1 * v2_avg + p_avg * normal_direction[2]
    f4 = f1 * v3_avg + p_avg * normal_direction[3]
    f5 = gammamean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    return SVector(f1, f2, f3, f4, f5)
end

@inline function flux_theta(u_ll, u_rr, orientation::Integer,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)

_, _, _, _, rho_theta_ll = u_ll
_, _, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = ln_mean(rho_ll, rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
v3_avg = 0.5f0 * (v3_ll + v3_rr)
p_avg = 0.5f0 * (p_ll + p_rr)

if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
    f4 = f1 * v3_avg
    f5 = gammamean * v1_avg
elseif orientation == 2
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
    f4 = f1 * v3_avg
    f5 = gammamean * v2_avg
else
    f1 = rho_mean * v3_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg
    f4 = f1 * v3_avg + p_avg
    f5 = gammamean * v3_avg
end

return SVector(f1, f2, f3, f4, f5)
end

# Entropy stable, density and pressure positivity preserving flux
@inline function flux_theta_es(u_ll, u_rr, orientation::Integer,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
    _, v1_ll, v2_ll, v3_ll, _ = cons2prim(u_ll, equations)
    _, v1_rr, v2_rr, v3_rr, _ = cons2prim(u_rr, equations)

    f_ec = flux_theta(u_ll, u_rr, orientation, equations)
    if orientation == 1
    lambda = max(abs(v1_ll), abs(v1_rr)) 
    elseif orientation == 2
        lambda = max(abs(v2_ll), abs(v2_rr))
    else
        lambda = max(abs(v3_ll), abs(v3_rr))
    end
    return f_ec - 0.5f0*lambda*(u_rr- u_ll)
end

# Entropy stable, density and pressure positivity preserving flux
@inline function flux_theta_es(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
    _, v1_ll, v2_ll, v3_ll, _ = cons2prim(u_ll, equations)
    _, v1_rr, v2_rr, v3_rr, _ = cons2prim(u_rr, equations)

    f_ec = flux_theta(u_ll, u_rr, normal_direction, equations)

    lambda = max(abs(v1_ll), abs(v1_rr))*normal_direction[1] + max(abs(v2_ll), abs(v2_rr))*normal_direction[2] + max(abs(v3_ll), abs(v3_rr))*normal_direction[3]

    return f_ec - 0.5f0*lambda*(u_rr- u_ll)
end

@inline function flux_theta_AM(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)
v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] + v3_ll * normal_direction[3]
v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] + v3_rr * normal_direction[3]
_, _, _, _, rho_theta_ll = u_ll
_, _, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = 0.5f0*(rho_ll + rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
v3_avg = 0.5f0 * (V3_ll + v3_rr)
p_avg = 0.5f0 * (p_ll + p_rr)

# Calculate fluxes depending on normal_direction
f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
f2 = f1 * v1_avg + p_avg * normal_direction[1]
f3 = f1 * v2_avg + p_avg * normal_direction[2]
f4 = f1 * v3_avg + p_avg * normal_direction[3]
f5 = gammamean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
return SVector(f1, f2, f3, f4, f5)
end

@inline function flux_theta_AM(u_ll, u_rr, orientation::Integer,
equations::CompressibleEulerPotentialTemperatureEquations3D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)

_, _, _, _, rho_theta_ll = u_ll
_, _, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = 0.5f0*(rho_rr + rho_ll)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
v3_avg = 0.5f0 * (v3_ll + v3_rr)
p_avg = 0.5f0 * (p_ll + p_rr)

if orientation == 1
f1 = rho_mean * v1_avg
f2 = f1 * v1_avg + p_avg
f3 = f1 * v2_avg
f4 = f1 * v3_avg
f5 = gammamean * v1_avg
elseif orientation == 2
f1 = rho_mean * v2_avg
f2 = f1 * v1_avg
f3 = f1 * v2_avg + p_avg
f4 = f1 * v3_avg
f5 = gammamean * v2_avg
else
f1 = rho_mean * v3_avg
f2 = f1 * v1_avg
f3 = f1 * v2_avg
f4 = f1 * v3_avg + p_avg
f5 = gammamean * v3_avg
end

return SVector(f1, f2, f3, f4, f5)
end

## Entropy (total energy) conservative flux for the Compressible Euler with the Potential Formulation

@inline function flux_theta_rhos(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)
v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] + v3_ll * normal_direction[3]
v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] + v3_rr * normal_direction[3]
_, _, _, _, rho_theta_ll = u_ll
_, _, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = ln_mean(rho_ll, rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
v3_avg = 0.5f0 * (V3_ll + v3_rr)
p_avg = 0.5f0 * (p_ll + p_rr)


# Calculate fluxes depending on normal_direction
f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
f2 = f1 * v1_avg + p_avg * normal_direction[1]
f3 = f1 * v2_avg + p_avg * normal_direction[2]
f4 = f1 * v3_avg + p_avg * normal_direction[3]
f5 = f1 * inv_ln_mean(rho_ll/ rho_theta_ll, rho_rr/rho_theta_rr)
return SVector(f1, f2, f3, f4, f5)
end

@inline function flux_theta_rhos(u_ll, u_rr, orientation::Integer,
equations::CompressibleEulerPotentialTemperatureEquations3D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)

_, _, _, _, rho_theta_ll = u_ll
_, _, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = ln_mean(rho_ll, rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
v3_avg = 0.5f0 * (v3_ll + v3_rr)
p_avg = 0.5f0 * (p_ll + p_rr)

if orientation == 1
f1 = rho_mean * v1_avg
f2 = f1 * v1_avg + p_avg
f3 = f1 * v2_avg
f4 = f1 * v3_avg
f5 = f1 * inv_ln_mean(rho_ll/ rho_theta_ll, rho_rr/rho_theta_rr)
elseif orientation == 2
f1 = rho_mean * v2_avg
f2 = f1 * v1_avg
f3 = f1 * v2_avg + p_avg
f4 = f1 * v3_avg
f5 = f1 * inv_ln_mean(rho_ll/ rho_theta_ll, rho_rr/rho_theta_rr)
else
f1 = rho_mean * v3_avg
f2 = f1 * v1_avg
f3 = f1 * v2_avg
f4 = f1 * v3_avg + p_avg
f5 = f1 * inv_ln_mean(rho_ll/ rho_theta_ll, rho_rr/rho_theta_rr)
end

return SVector(f1, f2, f3, f4, f5)
end

## Entropy (total energy) conservative flux for the Compressible Euler with the Potential Formulation

@inline function flux_theta_rhos_AM(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)
v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] + v3_ll * normal_direction[3]
v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] + v3_rr * normal_direction[3]
_, _, _, _, rho_theta_ll = u_ll
_, _, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = 0.5f0*(rho_ll + rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
v3_avg = 0.5f0 * (V3_ll + v3_rr)
p_avg = 0.5f0 * (p_ll + p_rr)


# Calculate fluxes depending on normal_direction
f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
f2 = f1 * v1_avg + p_avg * normal_direction[1]
f3 = f1 * v2_avg + p_avg * normal_direction[2]
f4 = f1 * v3_avg + p_avg * normal_direction[3]
f5 = f1 * inv_ln_mean(rho_ll/ rho_theta_ll, rho_rr/rho_theta_rr)
return SVector(f1, f2, f3, f4, f5)
end

@inline function flux_theta_rhos_AM(u_ll, u_rr, orientation::Integer,
equations::CompressibleEulerPotentialTemperatureEquations3D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)

_, _, _, _, rho_theta_ll = u_ll
_, _, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = 0.5f0*(rho_ll + rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
v3_avg = 0.5f0 * (v3_ll + v3_rr)
p_avg = 0.5f0 * (p_ll + p_rr)

if orientation == 1
f1 = rho_mean * v1_avg
f2 = f1 * v1_avg + p_avg
f3 = f1 * v2_avg
f4 = f1 * v3_avg
f5 = f1 * inv_ln_mean(rho_ll/ rho_theta_ll, rho_rr/rho_theta_rr)
elseif orientation == 2
f1 = rho_mean * v2_avg
f2 = f1 * v1_avg
f3 = f1 * v2_avg + p_avg
f4 = f1 * v3_avg
f5 = f1 * inv_ln_mean(rho_ll/ rho_theta_ll, rho_rr/rho_theta_rr)
else
f1 = rho_mean * v3_avg
f2 = f1 * v1_avg
f3 = f1 * v2_avg
f4 = f1 * v3_avg + p_avg
f5 = f1 * inv_ln_mean(rho_ll/ rho_theta_ll, rho_rr/rho_theta_rr)
end

return SVector(f1, f2, f3, f4, f5)
end

## Entropy (total energy) conservative flux for the Compressible Euler with the Potential Formulation

@inline function flux_theta_global(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)
v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] + v3_ll * normal_direction[3]
v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] + v3_rr * normal_direction[3]
_, _, _, _, rho_theta_ll = u_ll
_, _, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = ln_mean(rho_ll, rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
v3_avg = 0.5f0 * (V3_ll + v3_rr)
p_avg = 0.5f0 * (p_ll + p_rr)

# Calculate fluxes depending on normal_direction
f5 = gammamean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
f1 = f5 * ln_mean(rho_ll/rho_theta_ll, rho_rr/rho_theta_rr)
f2 = f1 * v1_avg + p_avg * normal_direction[1]
f3 = f1 * v2_avg + p_avg * normal_direction[2]
f4 = f1 * v3_avg + p_avg * normal_direction[3]

return SVector(f1, f2, f3, f4, f5)
end

@inline function flux_theta_global(u_ll, u_rr, orientation::Integer,
equations::CompressibleEulerPotentialTemperatureEquations3D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)

_, _, _, _, rho_theta_ll = u_ll
_, _, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = ln_mean(rho_ll, rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
v3_avg = 0.5f0 * (v3_ll + v3_rr)
p_avg = 0.5f0 * (p_ll + p_rr)
theta_mean = ln_mean(rho_ll/rho_theta_ll, rho_rr/rho_theta_rr)
if orientation == 1
f5 = gammamean * v1_avg
f1 = theta_mean * v1_avg
f2 = f1 * v1_avg + p_avg
f3 = f1 * v2_avg
f4 = f1 * v3_avg
elseif orientation == 2
f5 = gammamean * v2_avg
f1 = theta_mean * v2_avg
f2 = f1 * v1_avg
f3 = f1 * v2_avg + p_avg
f4 = f1 * v3_avg
else
f5 = gammamean * v3_avg
f1 = theta_mean * v3_avg
f2 = f1 * v1_avg
f3 = f1 * v2_avg
f4 = f1 * v3_avg + p_avg
end

return SVector(f1, f2, f3, f4, f5)
end

@inline function prim2cons(prim,
                           equations::CompressibleEulerPotentialTemperatureEquations3D)
    rho, v1, v2, v3, p = prim
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_v3 = rho * v3
    rho_theta = (p / equations.K)^(1 / equations.gamma)
    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_theta)
end

@inline function cons2prim(u,
                           equations::CompressibleEulerPotentialTemperatureEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_theta = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    p = equations.K*rho_theta^equations.gamma
    return SVector(rho, v1, v2, v3, p)
end

@inline function cons2cons(u,
                           equations::CompressibleEulerPotentialTemperatureEquations3D)
    return u
end

@inline function cons2entropy(u,
                              equations::CompressibleEulerPotentialTemperatureEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_theta = u

    w1 = -0.5f0 *(rho_v1^2 + rho_v2^2  + rho_v3^2)/rho^2
    w2 = rho_v1 / rho
    w3 = rho_v2 / rho
    w4 = rho_v3 / rho
    w5 = equations.gamma * equations.inv_gamma_minus_one * equations.K * (rho_theta)^(equations.gamma - 1)

    return SVector(w1, w2, w3, w4, w5)
end

@inline function cons2entropy2(u, equations::CompressibleEulerPotentialTemperatureEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_theta = u

    w1 = log(equations.K*(rho_theta/rho)^equations.gamma) - equations.gamma 
    w2 = 0.0
    w3 = 0.0
    w4 = 0.0
    w5 = rho/rho_theta*equations.gamma

    return SVector(w1, w2, w3, w4, w5)
end

@inline function entropy_math(cons,
                              equations::CompressibleEulerPotentialTemperatureEquations3D)
    # Mathematical entropy
    p = equations.K * cons[5]^equations.gamma
    U = (p / (equations.gamma - 1) + 0.5f0 * (cons[2]^2 + cons[3]^2 + cons[4]^2) / (cons[1]))

    return U
end


@inline function entropy_phys(cons, equations::CompressibleEulerPotentialTemperatureEquations3D)

    p = equations.K * cons[5]^equations.gamma
    # Thermodynamic entropy
    s = log(p) - equations.gamma * log(cons[1])
    S = -s*cons[1]/(equations.gamma-1.0)
    return S
end

# Default entropy is the mathematical entropy
@inline function entropy(cons,
                         equations::CompressibleEulerPotentialTemperatureEquations3D)
    entropy_math(cons, equations)
end

@inline function energy_total(cons,
                              equations::CompressibleEulerPotentialTemperatureEquations3D)
    entropy(cons, equations)
end

@inline function energy_kinetic(cons,
                                equations::CompressibleEulerPotentialTemperatureEquations3D)
    return 0.5f0 * (cons[2]^2 + cons[3]^2 + cons[4]^2) / (cons[1])
end

@inline function max_abs_speeds(u, equations::CompressibleEulerPotentialTemperatureEquations3D)
    rho, v1, v2, v3, p = cons2prim(u, equations)
    c = sqrt(equations.gamma * p / rho)

    return abs(v1) + c, abs(v2) + c, abs(v3) + c
end

@inline function density_pressure(u, equations::CompressibleEulerPotentialTemperatureEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_theta = u
    rho_times_p = rho*equations.p_0 * (equations.R * rho_theta / equations.p_0)^equations.gamma
    return rho_times_p
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)

# Get the velocity value in the appropriate direction
if orientation == 1
v_ll = v1_ll
v_rr = v1_rr
elseif orientation == 2
v_ll = v2_ll
v_rr = v2_rr
else # orientation == 3
v_ll = v3_ll
v_rr = v3_rr
end
# Calculate sound speeds
c_ll = sqrt(equations.gamma * p_ll / rho_ll)
c_rr = sqrt(equations.gamma * p_rr / rho_rr)

λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations3D)
rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)

# Calculate normal velocities and sound speed
# left
v_ll = (v1_ll * normal_direction[1]
+ v2_ll * normal_direction[2]
+ v3_ll * normal_direction[3])
c_ll = sqrt(equations.gamma * p_ll / rho_ll)
# right
v_rr = (v1_rr * normal_direction[1]
+ v2_rr * normal_direction[2]
+ v3_rr * normal_direction[3])
c_rr = sqrt(equations.gamma * p_rr / rho_rr)

return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr) * norm(normal_direction)
end

@inline function pressurecompute(cons,equations::CompressibleEulerPotentialTemperatureEquations3D)
    _,_,_,_,p = cons2prim(cons,equations)
    return p

end

end # @muladd
