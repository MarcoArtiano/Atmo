using Trixi
using Trixi: ln_mean, stolarsky_mean
import Trixi: varnames, cons2cons, cons2prim, cons2entropy, entropy, FluxLMARS

@muladd begin
#! format: noindent
struct CompressibleEulerPotentialTemperatureEquations2D{RealT <: Real} <:
       AbstractCompressibleEulerPotentialTemperatureEquations{2, 4}
    p_0::RealT
    c_p::RealT
    c_v::RealT
    g::RealT
    R::RealT
    gamma::RealT
    inv_gamma_minus_one::RealT
end

function CompressibleEulerPotentialTemperatureEquations2D(; g = 9.81, RealT = Float64)
    p_0 = 100_000.0
    c_p = 1004.0
    c_v = 717.0
    R = c_p - c_v
    gamma = c_p / c_v
    inv_gamma_minus_one = inv(gamma-1.0)
    return CompressibleEulerPotentialTemperatureEquations2D{RealT}(p_0, c_p, c_v, g, R,
                                                                   gamma, inv_gamma_minus_one)
end

function varnames(::typeof(cons2cons),
                  ::CompressibleEulerPotentialTemperatureEquations2D)
    ("rho", "rho_v1", "rho_v2", "rho_theta")
end

varnames(::typeof(cons2prim), ::CompressibleEulerPotentialTemperatureEquations2D) = ("rho",
                                                                                     "v1",
                                                                                     "v2",
                                                                                     "p1")

# Calculate 1D flux for a single point in the normal direction.
# Note, this directional vector is not normalized.
@inline function flux(u, normal_direction::AbstractVector,
                      equations::CompressibleEulerPotentialTemperatureEquations2D)
    rho, rho_v1, rho_v2, rho_theta = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
    rho_v_normal = rho * v_normal
    f1 = rho_v_normal
    p = equations.p_0 * (equations.R * rho_theta / equations.p_0)^equations.gamma

    f2 = (rho_v_normal) * v1 + p * normal_direction[1]
    f3 = (rho_v_normal) * v2 + p * normal_direction[2]
    f4 = (rho_theta) * v_normal
    return SVector(f1, f2, f3, f4)
end

@inline function flux(u, orientation::Integer,
                      equations::CompressibleEulerPotentialTemperatureEquations2D)
    rho, rho_v1, rho_v2, rho_theta = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = equations.p_0 * (equations.R * rho_theta / equations.p_0)^equations.gamma

    if orientation == 1
        f1 = rho_v1
        f2 = rho_v1 * v1 + p
        f3 = rho_v1 * v2
        f4 = rho_theta * v1
    else
        f1 = rho_v2
        f2 = rho_v2 * v1
        f3 = rho_v2 * v2 + p
        f4 = rho_theta * v2
    end

    return SVector(f1, f2, f3, f4)
end

# Slip-wall boundary condition
# Determine the boundary numerical surface flux for a slip wall condition.
# Imposes a zero normal velocity at the wall.
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerPotentialTemperatureEquations2D)
    @unpack gamma = equations
    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal = normal_direction / norm_

    # rotate the internal solution state
    u_local = rotate_to_x(u_inner, normal, equations)

    # compute the primitive variables
    rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)
    # Get the solution of the pressure Riemann problem
    # See Section 6.3.3 of
    # Eleuterio F. Toro (2009)
    # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
    # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
    if v_normal <= 0.0
        sound_speed = sqrt(gamma * p_local / rho_local) # local sound speed
        p_star = p_local *
                 (1.0 + 0.5f0 * (gamma - 1) * v_normal / sound_speed)^(2.0 * gamma *equations.inv_gamma_minus_one)
    else # v_normal > 0.0
        A = 2.0 / ((gamma + 1) * rho_local)
        B = p_local * (gamma - 1) / (gamma + 1)
        p_star = p_local +
                 0.5f0 * v_normal / A *
                 (v_normal + sqrt(v_normal^2 + 4.0 * A * (p_local + B)))
    end

    # For the slip wall we directly set the flux as the normal velocity is zero
    return SVector(zero(eltype(u_inner)),
                   p_star * normal[1],
                   p_star * normal[2],
                   zero(eltype(u_inner))) * norm_
end

# Fix sign for structured mesh.
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerPotentialTemperatureEquations2D)
    # flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
    # to be inward pointing on the -x and -y sides due to the orientation convention used by StructuredMesh
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
                                      equations::CompressibleEulerPotentialTemperatureEquations2D)
    rho, _, _, _ = u
    return SVector(zero(eltype(u)), zero(eltype(u)), -equations.g * rho,
                   zero(eltype(u)))
end


@inline function initial_condition_density_wave(x, t, equations::CompressibleEulerPotentialTemperatureEquations2D)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    rho = 1 + convert(RealT, 0.98) * sinpi(2 * (x[1] + x[2] - t * (v1 + v2)))
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 20
    rho_e = p / (equations.gamma - 1) + 0.5f0 * rho * (v1^2 + v2^2)
    return prim2cons(SVector(rho, v1, v2, p),equations)
end

# Rotate momentum flux. The same as in compressible Euler.
@inline function rotate_to_x(u, normal_vector::AbstractVector,
                             equations::CompressibleEulerPotentialTemperatureEquations2D)
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D rotation matrix with normal and tangent directions of the form
    # [ 1    0    0   0;
    #   0   n_1  n_2  0;
    #   0   t_1  t_2  0;
    #   0    0    0   1 ]
    # where t_1 = -n_2 and t_2 = n_1

    return SVector(u[1],
                   c * u[2] + s * u[3],
                   -s * u[2] + c * u[3],
                   u[4])
end

# Low Mach number approximate Riemann solver (LMARS) from
# X. Chen, N. Andronova, B. Van Leer, J. E. Penner, J. P. Boyd, C. Jablonowski, S.
# Lin, A Control-Volume Model of the Compressible Euler Equations with a Vertical Lagrangian
# Coordinate Monthly Weather Review Vol. 141.7, pages 2526–2544, 2013,
# https://journals.ametsoc.org/view/journals/mwre/141/7/mwr-d-12-00129.1.xml.

@inline function (flux_lmars::FluxLMARS)(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::CompressibleEulerPotentialTemperatureEquations2D)
    a = flux_lmars.speed_of_sound
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    norm_ = norm(normal_direction)

    rho = 0.5f0 * (rho_ll + rho_rr)

    p_interface = 0.5f0 * (p_ll + p_rr) - 0.5f0 * a * rho * (v_rr - v_ll) / norm_
    v_interface = 0.5f0 * (v_ll + v_rr) - 1 / (2 * a * rho) * (p_rr - p_ll) * norm_

    if (v_interface > 0)
        f1, f2, f3, f4 = u_ll * v_interface
    else
        f1, f2, f3, f4 = u_rr * v_interface
    end

    return SVector(f1,
                   f2 + p_interface * normal_direction[1],
                   f3 + p_interface * normal_direction[2],
                   f4)
end

## Entropy (total energy) conservative flux for the Compressible Euler with the Potential Formulation

@inline function flux_theta(u_ll, u_rr, normal_direction::AbstractVector,
                            equations::CompressibleEulerPotentialTemperatureEquations2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    _, _, _, rho_theta_ll = u_ll
    _, _, _, rho_theta_rr = u_rr
    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)

    gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    f2 = f1 * v1_avg + p_avg * normal_direction[1]
    f3 = f1 * v2_avg + p_avg * normal_direction[2]
    f4 = gammamean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    return SVector(f1, f2, f3, f4)
end

@inline function (flux_lmars::FluxLMARS)(u_ll, u_rr, orientation::Integer,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
c = flux_lmars.speed_of_sound

# Unpack left and right state
rho_ll, v1_ll, v2_ll,  p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr,  p_rr = cons2prim(u_rr, equations)

if orientation == 1
v_ll = v1_ll
v_rr = v1_rr
else orientation == 2
v_ll = v2_ll
v_rr = v2_rr
end

rho = 0.5 * (rho_ll + rho_rr)
p = 0.5 * (p_ll + p_rr) - 0.5 * c * rho * (v_rr - v_ll)
v = 0.5 * (v_ll + v_rr) - 1 / (2 * c * rho) * (p_rr - p_ll)

# We treat the energy term analogous to the potential temperature term in the paper by
# Chen et al., i.e. we use p_ll and p_rr, and not p
if v >= 0
f1, f2, f3, f4 = v * u_ll
else
f1, f2, f3, f4 = v * u_rr
end

if orientation == 1
f2 += p
else
f3 += p
end

return SVector(f1, f2, f3, f4)
end


@inline function flux_theta_AM(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)
v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
_, _, _, rho_theta_ll = u_ll
_, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = 0.5*(rho_ll + rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
p_avg = 0.5f0 * (p_ll + p_rr)

# Calculate fluxes depending on normal_direction
f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
f2 = f1 * v1_avg + p_avg * normal_direction[1]
f3 = f1 * v2_avg + p_avg * normal_direction[2]
f4 = gammamean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
return SVector(f1, f2, f3, f4)
end

@inline function flux_theta(u_ll, u_rr, orientation::Integer,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

_, _, _, rho_theta_ll = u_ll
_, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = ln_mean(rho_ll, rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
p_avg = 0.5f0 * (p_ll + p_rr)

if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
    f4 = gammamean * v1_avg
else 
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
    f4 = gammamean * v2_avg
end

return SVector(f1, f2, f3, f4)
end

@inline function flux_theta_AM(u_ll, u_rr, orientation::Integer,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

_, _, _, rho_theta_ll = u_ll
_, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = 0.5*(rho_ll + rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
p_avg = 0.5f0 * (p_ll + p_rr)

if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
    f4 = gammamean * v1_avg
else 
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
    f4 = gammamean * v2_avg
end

return SVector(f1, f2, f3, f4)
end

# Entropy stable, density and pressure positivity preserving flux
@inline function flux_theta_es(u_ll, u_rr, orientation::Integer,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
    _, v1_ll, v2_ll, _ = cons2prim(u_ll, equations)
    _, v1_rr, v2_rr, _ = cons2prim(u_rr, equations)

    f_ec = flux_theta(u_ll, u_rr, orientation, equations)
    if orientation == 1
    lambda = max(abs(v1_ll), abs(v1_rr)) 
    else
        lambda = max(abs(v2_ll), abs(v2_rr))
    end
    return f_ec - 0.5f0*lambda*(u_rr- u_ll)
end

# Entropy stable, density and pressure positivity preserving flux
@inline function flux_theta_es(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
    _, v1_ll, v2_ll, _ = cons2prim(u_ll, equations)
    _, v1_rr, v2_rr, _ = cons2prim(u_rr, equations)

    f_ec = flux_theta(u_ll, u_rr, normal_direction, equations)

    lambda = max(abs(v1_ll), abs(v1_rr))*normal_direction[1] + max(abs(v2_ll), abs(v2_rr))*normal_direction[2]

    return f_ec - 0.5f0*lambda*(u_rr- u_ll)
end

@inline function flux_theta_rhoAM(u_ll, u_rr, normal_direction::AbstractVector,
                                  equations::CompressibleEulerPotentialTemperatureEquations2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    _, _, _, rho_theta_ll = u_ll
    _, _, _, rho_theta_rr = u_rr
    # Compute the necessary mean values
    rho_mean = 0.5f0 * (rho_ll + rho_rr)

    gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    f2 = f1 * v1_avg + p_avg * normal_direction[1]
    f3 = f1 * v2_avg + p_avg * normal_direction[2]
    f4 = gammamean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    return SVector(f1, f2, f3, f4)
end

@inline function flux_theta_global(u_ll, u_rr, normal_direction::AbstractVector,
                                        equations::CompressibleEulerPotentialTemperatureEquations2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    _, _, _, rho_theta_ll = u_ll
    _, _, _, rho_theta_rr = u_rr
    # Compute the necessary mean values
    #rho_mean = ln_mean(rho_ll, rho_rr)
    #rho_mean = 0.5f0 * (rho_ll + rho_rr)
    rho_mean = ln_mean(rho_ll / rho_theta_ll, rho_rr / rho_theta_rr)
    gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)

    # Calculate fluxes depending on normal_direction
    #f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    f4 = gammamean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    f1 = f4 * ln_mean(rho_ll/rho_theta_ll, rho_rr/rho_theta_rr)
    f2 = f1 * v1_avg + p_avg * normal_direction[1]
    f3 = f1 * v2_avg + p_avg * normal_direction[2]

    return SVector(f1, f2, f3, f4)
end


@inline function flux_theta_global(u_ll, u_rr, orientation::Integer,
equations::CompressibleEulerPotentialTemperatureEquations2D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

_, _, _, rho_theta_ll = u_ll
_, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = ln_mean(rho_ll, rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
p_avg = 0.5f0 * (p_ll + p_rr)
theta_mean = ln_mean(rho_ll/rho_theta_ll, rho_rr/rho_theta_rr)
if orientation == 1
f4 = gammamean * v1_avg
f1 = theta_mean * v1_avg
f2 = f1 * v1_avg + p_avg
f3 = f1 * v2_avg
elseif orientation == 2
f4 = gammamean * v2_avg
f1 = theta_mean * v2_avg
f2 = f1 * v1_avg
f3 = f1 * v2_avg + p_avg
end

return SVector(f1, f2, f3, f4)
end

@inline function flux_theta_rhos(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)
v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
_, _, _, rho_theta_ll = u_ll
_, _, _, rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = ln_mean(rho_ll, rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
p_avg = 0.5f0 * (p_ll + p_rr)


# Calculate fluxes depending on normal_direction
f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
f2 = f1 * v1_avg + p_avg * normal_direction[1]
f3 = f1 * v2_avg + p_avg * normal_direction[2]
f4 = f1 * inv_ln_mean(rho_ll/ rho_theta_ll, rho_rr/rho_theta_rr)
return SVector(f1, f2, f3, f4)
end

@inline function flux_theta_rhos(u_ll, u_rr, orientation::Integer,
equations::CompressibleEulerPotentialTemperatureEquations2D)
# Unpack left and right state
rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

_, _, _,  rho_theta_ll = u_ll
_, _, _,  rho_theta_rr = u_rr
# Compute the necessary mean values
rho_mean = ln_mean(rho_ll, rho_rr)

gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

v1_avg = 0.5f0 * (v1_ll + v1_rr)
v2_avg = 0.5f0 * (v2_ll + v2_rr)
p_avg = 0.5f0 * (p_ll + p_rr)

if orientation == 1
f1 = rho_mean * v1_avg
f2 = f1 * v1_avg + p_avg
f3 = f1 * v2_avg
f4 = f1 * inv_ln_mean(rho_ll/ rho_theta_ll, rho_rr/rho_theta_rr)
else
f1 = rho_mean * v2_avg
f2 = f1 * v1_avg
f3 = f1 * v2_avg + p_avg
f4 = f1 * inv_ln_mean(rho_ll/ rho_theta_ll, rho_rr/rho_theta_rr)
end

return SVector(f1, f2, f3, f4)
end

@inline function flux_theta_global_es(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
    _, v1_ll, v2_ll, _ = cons2prim(u_ll, equations)
    _, v1_rr, v2_rr, _ = cons2prim(u_rr, equations)

    f_ec = flux_theta_global(u_ll, u_rr, normal_direction, equations)

    lambda = max(abs(v1_ll), abs(v1_rr))*normal_direction[1] + max(abs(v2_ll), abs(v2_rr))*normal_direction[2]

    return f_ec - 0.5f0*lambda/340.0*(u_rr- u_ll)
end

@inline function flux_theta_physentropy2(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::CompressibleEulerPotentialTemperatureEquations2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    _, _, _, rho_theta_ll = u_ll
    _, _, _, rho_theta_rr = u_rr
    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)
    #rho_mean = 0.5f0 * (rho_ll + rho_rr)
    #rho_mean = ln_mean(rho_ll/rho_theta_ll, rho_rr/rho_theta_rr)
    gammamean = stolarsky_mean(rho_theta_ll, rho_theta_rr, equations.gamma)

    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    f2 = f1 * v1_avg + p_avg * normal_direction[1]
    f3 = f1 * v2_avg + p_avg * normal_direction[2]
    f4 = inv_ln_mean(rho_ll / rho_theta_ll, rho_rr / rho_theta_rr) * f1
    return SVector(f1, f2, f3, f4)
end

@inline function splitting_vanleer_haenel(u, orientation_or_normal_direction,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
fm = splitting_vanleer_haenel(u, Val{:minus}(), orientation_or_normal_direction,
equations)
fp = splitting_vanleer_haenel(u, Val{:plus}(), orientation_or_normal_direction,
equations)
return fm, fp
end

@inline function splitting_vanleer_haenel(u, ::Val{:plus}, orientation::Integer,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
rho, rho_v1, rho_v2, rho_theta = u
v1 = rho_v1 / rho
v2 = rho_v2 / rho
K = equations.p_0 * (equations.R  / equations.p_0)^equations.gamma
p = equations.p_0 * (equations.R * rho_theta / equations.p_0)^equations.gamma

a = sqrt(equations.gamma * p / rho)

if orientation == 1
M = v1 / a
p_plus = 0.5f0 * (1 + equations.gamma * M) * p

f1p = 0.25f0 * rho * a * (M + 1)^2
f2p = f1p * v1 + p_plus
f3p = f1p * v2
f4p = 0.25f0 * (M + 1)^2 * a * (rho * a^2/(K*equations.gamma))^(1/equations.gamma)
else # orientation == 2
M = v2 / a
p_plus = 0.5f0 * (1 + equations.gamma * M) * p

f1p = 0.25f0 * rho * a * (M + 1)^2
f2p = f1p * v1
f3p = f1p * v2 + p_plus
f4p = 0.25f0 * (M + 1)^2 * a * (rho * a^2/(K*equations.gamma))^(1/equations.gamma)
end
return SVector(f1p, f2p, f3p, f4p)
end

@inline function splitting_vanleer_haenel(u, ::Val{:minus}, orientation::Integer,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
rho, rho_v1, rho_v2, rho_theta = u
v1 = rho_v1 / rho
v2 = rho_v2 / rho
K = equations.p_0 * (equations.R  / equations.p_0)^equations.gamma
p = equations.p_0 * (equations.R * rho_theta / equations.p_0)^equations.gamma

a = sqrt(equations.gamma * p / rho)

if orientation == 1
M = v1 / a
p_minus = 0.5f0 * (1 - equations.gamma * M) * p

f1m = -0.25f0 * rho * a * (M - 1)^2
f2m = f1m * v1 + p_minus
f3m = f1m * v2
f4m = -0.25f0 * (M - 1)^2 * a * (rho * a^2/(K*equations.gamma))^(1/equations.gamma)
else # orientation == 2
M = v2 / a
p_minus = 0.5f0 * (1 - equations.gamma * M) * p

f1m = -0.25f0 * rho * a * (M - 1)^2
f2m = f1m * v1
f3m = f1m * v2 + p_minus
f4m = -0.25f0 * (M - 1)^2 * a * (rho * a^2/(K*equations.gamma))^(1/equations.gamma)
end
return SVector(f1m, f2m, f3m, f4m)
end

@inline function splitting_vanleer_haenel(u, ::Val{:plus},
    normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
rho, rho_v1, rho_v2, rho_theta = u
v1 = rho_v1 / rho
v2 = rho_v2 / rho
K = equations.p_0 * (equations.R  / equations.p_0)^equations.gamma
p = equations.p_0 * (equations.R * rho_theta / equations.p_0)^equations.gamma

a = sqrt(equations.gamma * p / rho)


v_n = normal_direction[1] * v1 + normal_direction[2] * v2
M = v_n / a
p_plus = 0.5f0 * (1 + equations.gamma * M) * p

f1p = 0.25f0 * rho * a * (M + 1)^2
f2p = f1p * v1 + normal_direction[1] * p_plus
f3p = f1p * v2 + normal_direction[2] * p_plus
f4p = 0.25f0 * (M + 1)^2 * (rho * a^2/(K*equations.gamma))^(1/equations.gamma)*a

return SVector(f1p, f2p, f3p, f4p)
end

@inline function splitting_vanleer_haenel(u, ::Val{:minus},
    normal_direction::AbstractVector,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
rho, rho_v1, rho_v2, rho_theta = u
v1 = rho_v1 / rho
v2 = rho_v2 / rho
K = equations.p_0 * (equations.R  / equations.p_0)^equations.gamma
p = equations.p_0 * (equations.R * rho_theta / equations.p_0)^equations.gamma

a = sqrt(equations.gamma * p / rho)

v_n = normal_direction[1] * v1 + normal_direction[2] * v2
M = v_n / a
p_minus = 0.5f0 * (1 - equations.gamma * M) * p

f1m = -0.25f0 * rho * a * (M - 1)^2
f2m = f1m * v1 + normal_direction[1] * p_minus
f3m = f1m * v2 + normal_direction[2] * p_minus
f4m = -0.25f0 * (M - 1)^2 * (rho * a^2/(K*equations.gamma))^(1/equations.gamma)*a

return SVector(f1m, f2m, f3m, f4m)
end


@inline function prim2cons(prim,
                           equations::CompressibleEulerPotentialTemperatureEquations2D)
    rho, v1, v2, p = prim
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_theta = (p / equations.p_0)^(1 / equations.gamma) * equations.p_0 / equations.R
    return SVector(rho, rho_v1, rho_v2, rho_theta)
end

@inline function cons2prim(u,
                           equations::CompressibleEulerPotentialTemperatureEquations2D)
    rho, rho_v1, rho_v2, rho_theta = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = equations.p_0 * (equations.R * rho_theta / equations.p_0)^equations.gamma

    return SVector(rho, v1, v2, p)
end

@inline function cons2cons(u,
                           equations::CompressibleEulerPotentialTemperatureEquations2D)
    return u
end

@inline function cons2entropy(u,
                              equations::CompressibleEulerPotentialTemperatureEquations2D)
    rho, rho_v1, rho_v2, rho_theta = u

    k = equations.p_0 * (equations.R / equations.p_0)^equations.gamma
    w1 = -0.5f0 * rho_v1^2 / (rho)^2 - 0.5f0 * rho_v2^2 / (rho)^2
    w2 = rho_v1 / rho
    w3 = rho_v2 / rho
    w4 = equations.gamma / (equations.gamma - 1) * k * (rho_theta)^(equations.gamma - 1)

    return SVector(w1, w2, w3, w4)
end

@inline function entropy_math(cons,
                              equations::CompressibleEulerPotentialTemperatureEquations2D)
    # Mathematical entropy
    p = equations.p_0 * (equations.R * cons[4] / equations.p_0)^equations.gamma

    U = (p / (equations.gamma - 1) + 0.5f0 * (cons[2]^2 + cons[3]^2) / (cons[1]))

    return U
end

# Default entropy is the mathematical entropy
@inline function entropy(cons,
                         equations::CompressibleEulerPotentialTemperatureEquations2D)
    entropy_math(cons, equations)
end

@inline function energy_total(cons,
                              equations::CompressibleEulerPotentialTemperatureEquations2D)
    entropy(cons, equations)
end

@inline function energy_kinetic(cons,
                                equations::CompressibleEulerPotentialTemperatureEquations2D)
    return 0.5f0 * (cons[2]^2 + cons[3]^2) / (cons[1])
end

@inline function max_abs_speeds(u,
                                equations::CompressibleEulerPotentialTemperatureEquations2D)
    rho, v1, v2, p = cons2prim(u, equations)
    c = sqrt(equations.gamma * p / rho)

    return abs(v1) + c, abs(v2) + c
end

@inline function density_pressure(u, equations::CompressibleEulerPotentialTemperatureEquations2D)
    rho, rho_v1, rho_v2, rho_theta = u
    rho_times_p = rho*equations.p_0 * (equations.R * rho_theta / equations.p_0)^equations.gamma
    return rho_times_p
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
    equations::CompressibleEulerPotentialTemperatureEquations2D)
rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

# Get the velocity value in the appropriate direction
if orientation == 1
v_ll = v1_ll
v_rr = v1_rr
else # orientation == 2
v_ll = v2_ll
v_rr = v2_rr
end
# Calculate sound speeds
c_ll = sqrt(equations.gamma * p_ll / rho_ll)
c_rr = sqrt(equations.gamma * p_rr / rho_rr)

λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

end # @muladd
