# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
	LinearizedEulerEquations2D(v_mean_global, c_mean_global, rho_mean_global)

Linearized Euler equations in two space dimensions. The equations are given by
```math
\partial_t
\begin{pmatrix}
	\v_1' \\ v_2' \\ p' \\ b'
\end{pmatrix}
+
\partial_x
\begin{pmatrix}
	p \\ 0 \\ cs^2 v_1 \\ 0
\end{pmatrix}
+
\partial_z
\begin{pmatrix}
	0 \\ p \\ cs^2 v_2 \\ 0
\end{pmatrix}
=
\begin{pmatrix}
	0 \\ b \\ 0 \\ -N^2 v_2
\end{pmatrix}
```
The bar ``\bar{(\cdot)}`` indicates uniform mean flow variables and ``c`` is the speed of sound.
The unknowns are the acoustic velocities ``v' = (v_1', v_2')``, the pressure ``p'`` and the density ``\rho'``.
"""
struct LinearizedGravityWaveEquations2D{RealT <: Real} <:
	   AbstractLinearizedEulerEquations{2, 4}
	cs::RealT  # speed of sound
	U::RealT
	fb::RealT # Buoyancy frequency
end

function LinearizedGravityWaveEquations2D(cs::Real, U::Real, fb::Real)
	return LinearizedGravityWaveEquations2D(cs, U, fb)
end

function varnames(::typeof(cons2cons), ::LinearizedGravityWaveEquations2D)
	("u", "w", "p", "b")
end
function varnames(::typeof(cons2prim), ::LinearizedGravityWaveEquations2D)
	("u", "w", "p", "b")
end

"""
	initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations2D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t,
	equations::LinearizedGravityWaveEquations2D)
	A = 5000
	H = 10000
	b0 = 0.01
	xc = 0.0
	binv = (1 + (x[1] - xc)^2 / A^2)
	b = b0 * sin(pi * x[2] / H) / binv

	return SVector(0.0, 0.0, 0.0, b)
end

"""
boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
surface_flux_function, equations::ShallowWaterEquations2D)

Should be used together with [`TreeMesh`](@ref).
"""


@inline function boundary_condition_slip_wall(u_inner, orientation,
	direction, x, t,
	surface_flux_function,
	equations::LinearizedGravityWaveEquations2D)
## get the appropriate normal vector from the orientation
if orientation == 1
u_boundary = SVector(-u_inner[1], u_inner[2], u_inner[3], u_inner[4])
else # orientation == 2
u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4])
end
#	surface_flux_function = flux_lmars2_no_advection
# Calculate boundary flux
if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
else # u_boundary is "left" of boundary, u_inner is "right" of boundary
flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
end

return flux
end

@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
	x, t,
	surface_flux_function,
	equations::LinearizedGravityWaveEquations2D)
# normalize the outward pointing direction
normal = normal_direction / norm(normal_direction)

# compute the normal velocity
u_normal = normal[1] * u_inner[1] + normal[2] * u_inner[2]

# create the "external" boundary solution state
u_boundary = SVector(u_inner[1] - 2.0*u_normal*normal[1],
u_inner[2] - 2.0 * u_normal * normal[2],
u_inner[3],
u_inner[4])
#surface_flux_function = flux_lmars2_no_advection
# calculate the boundary flux
flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

return flux
end

@inline function rotate_to_x(u, normal_vector, equations::LinearizedGravityWaveEquations2D)
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

	return SVector(c * u[1] + s * u[2],
		-s * u[1] + c * u[2],
		u[3],
		u[4])
end


@inline function source_terms_convergence_test(u, x, t,
	equations::LinearizedGravityWaveEquations2D)
	# Same settings as in `initial_condition`
	@unpack fb = equations
	_, v2, _, b = u
	du1 = 0.0
	du2 = b
	du3 = 0
	du4 = -fb^2 * v2

	return SVector(du1, du2, du3, du4)
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer,
	equations::LinearizedGravityWaveEquations2D)
	@unpack cs, U = equations
	v1, v2, p, b = u
	if orientation == 1
		f1 = U*v1 + p
		f2 = U*v2
		f3 = U*p+ cs^2 * v1
		f4 = U*b
	else
		f1 = 0.0
		f2 = p
		f3 = cs^2 * v2
		f4 = 0.0
	end

	return SVector(f1, f2, f3, f4)
end

# Calculate 1D flux for a single point
@inline function flux(u, normal_direction::AbstractVector,
	equations::LinearizedGravityWaveEquations2D)
	@unpack cs, U = equations

	v1, v2, p, b = u

	v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]

	f1 = normal_direction[1] * p  + v1*U*normal_direction[1]
	f2 = normal_direction[2] * p +  v2*U*normal_direction[1]
	f3 = v_normal * cs^2 + U*p*normal_direction[1] 
	f4 = U*b*normal_direction[1]

	return SVector(f1, f2, f3, f4)
end

@inline have_constant_speed(::LinearizedGravityWaveEquations2D) = True()

@inline function max_abs_speeds(equations::LinearizedGravityWaveEquations2D)
	@unpack cs = equations
	return cs, cs
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
	equations::LinearizedGravityWaveEquations2D)
	@unpack v_mean_global, c_mean_global = equations
	if orientation == 1
		return abs(v_mean_global[1]) + c_mean_global
	else # orientation == 2
		return abs(v_mean_global[2]) + c_mean_global
	end
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
	equations::LinearizedGravityWaveEquations2D)
	@unpack v_mean_global, c_mean_global = equations
	v_mean_normal = normal_direction[1] * v_mean_global[1] +
					normal_direction[2] * v_mean_global[2]
	return abs(v_mean_normal) + c_mean_global * norm(normal_direction)
end


@inline function flux_lmars2(u_ll, u_rr, orientation::Integer,
	equations::LinearizedGravityWaveEquations2D)
	@unpack cs, U  = equations

	v1_ll, v2_ll, p_ll, b_ll = u_ll
	v1_rr, v2_rr, p_rr, b_rr = u_rr

	p1_int = (p_ll + p_rr) * 0.5 - 0.5 * cs * (v1_rr - v1_ll)
	p2_int = (p_ll + p_rr) * 0.5 - 0.5 * cs * (v2_rr - v2_ll)
	v1_int = (v1_ll + v1_rr) * 0.5 - 1 / (2 * cs) * (p_rr - p_ll)
	v2_int = (v2_ll + v2_rr) * 0.5 - 1 / (2 * cs) * (p_rr - p_ll)
	println("ciao")
	if orientation == 1
		f1 = p1_int  
		f2 = 0.0
		f3 = cs^2 * v1_int
		f4 = 0.0

		f1 = f1 + U*v1_int
		f2 = f2 + U*v2_int
		f3 = f3 + U*p_int
		f4 = f4 + U*(b_rr + b_ll)*0.5
		# advection
		# if v1_rr < 0
		# f1 = f1 + U*v1_rr
		# f2 = f2 + U*v2_rr
		# f3 = f3 + U*p_rr
		# f4 = f4 + U*b_rr
		# else
		# 	f1 = f1 + U*v1_ll
		# 	f2 = f2 + U*v2_ll
		# 	f3 = f3 + U*p_ll
		# 	f4 = f4 + U*b_ll
		# end	
	else # orientation == 2
		f1 = 0.0
		f2 = p2_int
		f3 = cs^2 * v2_int
		f4 = 0.0
	end
	return SVector(f1, f2, f3, f4)
end

@inline function flux_lmars2_no_advection(u_ll, u_rr, orientation::Integer,
	equations::LinearizedGravityWaveEquations2D)
	@unpack cs, U = equations

	v1_ll, v2_ll, p_ll, b_ll = u_ll
	v1_rr, v2_rr, p_rr, b_rr = u_rr

	p1_int = (p_ll + p_rr) * 0.5 - 0.5 * cs * (v1_rr - v1_ll)
	p2_int = (p_ll + p_rr) * 0.5 - 0.5 * cs * (v2_rr - v2_ll)
	v1_int = (v1_ll + v1_rr) * 0.5 - 1 / (2 * cs) * (p_rr - p_ll)
	v2_int = (v2_ll + v2_rr) * 0.5 - 1 / (2 * cs) * (p_rr - p_ll)

	if orientation == 1
		f1 = p1_int
		f2 = 0.0
		f3 = cs^2 * v1_int
		f4 = 0.0
	else # orientation == 2
		f1 = 0.0
		f2 = p2_int
		f3 = cs^2 * v2_int
		f4 = 0.0
	end
	return SVector(f1, f2, f3, f4)
end

@inline function flux_lmars2(u_ll, u_rr, normal_direction::AbstractVector,
	equations::LinearizedGravityWaveEquations2D)
	@unpack cs, U = equations

	v1_ll, v2_ll, p_ll, b_ll = u_ll
	v1_rr, v2_rr, p_rr, b_rr = u_rr
	norm_ = norm(normal_direction)
	normal_vector = normal_direction / norm_
    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    p_interface = 0.5 * (p_ll + p_rr) - 0.5 * cs * (v_rr - v_ll) / norm_
    v_interface = 0.5 * (v_ll + v_rr) - 1 / (2 * cs) * (p_rr - p_ll) * norm_
	v1_interface = 0.5 * (v1_ll + v1_rr) - 1 / (2 * cs) * (p_rr - p_ll)
	v2_interface = 0.5 * (v2_ll + v2_rr) - 1 / (2 * cs) * (p_rr - p_ll)
	b_avg = 0.5*(b_ll + b_rr)

	f1 = normal_direction[1] * p_interface# + U*v1_interface*normal_direction[1]
	f2 = normal_direction[2] * p_interface# + U*v2_interface*normal_direction[1]
	f3 = cs^2 * v_interface #+ U*p_interface*normal_direction[1]
	f4 = 0.0#U * b_avg*normal_direction[1]

	if normal_direction[1] < 0
		f1 = f1 + U*v1_rr*normal_direction[1]
		f2 = f2 + U*v2_rr*normal_direction[1]
		f3 = f3 + U*p_rr*normal_direction[1]
		f4 = f4 + U*b_rr*normal_direction[1]
	else
		f1 = f1 + U*v1_ll*normal_direction[1]
		f2 = f2 + U*v2_ll*normal_direction[1]
		f3 = f3 + U*p_ll*normal_direction[1]
		f4 = f4 + U*b_ll*normal_direction[1]
	end

	return SVector(f1, f2, f3, f4)
end


# Calculate estimate for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
	equations::LinearizedGravityWaveEquations2D)
	min_max_speed_davis(u_ll, u_rr, orientation, equations)
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
	equations::LinearizedGravityWaveEquations2D)
	min_max_speed_davis(u_ll, u_rr, normal_direction, equations)
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
	equations::LinearizedGravityWaveEquations2D)
	@unpack v_mean_global, c_mean_global = equations

	λ_min = v_mean_global[orientation] - c_mean_global
	λ_max = v_mean_global[orientation] + c_mean_global

	return λ_min, λ_max
end

@inline function min_max_speed_davis(u_ll, u_rr, normal_direction::AbstractVector,
	equations::LinearizedGravityWaveEquations2D)
	@unpack v_mean_global, c_mean_global = equations

	norm_ = norm(normal_direction)

	v_normal = v_mean_global[1] * normal_direction[1] +
			   v_mean_global[2] * normal_direction[2]

	# The v_normals are already scaled by the norm
	λ_min = v_normal - c_mean_global * norm_
	λ_max = v_normal + c_mean_global * norm_

	return λ_min, λ_max
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearizedGravityWaveEquations2D) = u
@inline cons2entropy(u, ::LinearizedGravityWaveEquations2D) = u
end # muladd
