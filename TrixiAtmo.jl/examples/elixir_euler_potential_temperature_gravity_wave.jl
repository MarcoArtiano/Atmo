using OrdinaryDiffEq
using Trixi, TrixiAtmo
using TrixiAtmo: source_terms_gravity

function prim2velocity(u, x, equations) 
    rho, rho_v1, rho_v2, rho_e = u

    v2 = rho_v2 / rho

    return v2
end

function get_velocity(sol, semi, equations, cells_per_dimension, polydeg)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u_wrap = Trixi.wrap_array(sol.u[end], semi)
    velocity = Array{Float64}(
        undef,
        polydeg + 1,
        cells_per_dimension[1],
    )
    x_mesh = Array{Float64}(
        undef,
        polydeg + 1,
        cells_per_dimension[1],
    )
    @unpack node_coordinates = cache.elements
    for element in 101:150
        for i in eachnode(solver)
            x_local =
                Trixi.get_node_coords(node_coordinates, equations, solver, i, 3, element)
            velocity[i,element-100] =
                prim2velocity(u_wrap[:, i, 3, element], x_local, equations)
                x_mesh[i,element-100] = x_local[1]

        end
    end

    return velocity, x_mesh

end

function initial_condition_gravity_wave(x, t,
                                        equations::CompressibleEulerPotentialTemperatureEquations2D)
    g = equations.g
    c_p = equations.c_p
    c_v = equations.c_v
    # center of perturbation
    x_c = 100_000.0
    a = 5_000
    H = 10_000
    # perturbation in potential temperature
    potential_temperature_ref = 300.0 * exp(0.01^2 / g * x[2])
    potential_temperature_perturbation = 0.01 * sinpi(x[2] / H) / (1 + (x[1] - x_c)^2 / a^2)
    potential_temperature = potential_temperature_ref + potential_temperature_perturbation

    # Exner pressure, solves hydrostatic equation for x[2]
    exner = 1 + g^2 / (c_p * 300.0 * 0.01^2) * (exp(-0.01^2 / g * x[2]) - 1)

    # pressure
    p_0 = 100_000.0  # reference pressure
    R = c_p - c_v    # gas constant (dry air)
    p = p_0 * exner^(c_p / R)
    T = potential_temperature * exner

    # density
    rho = p / (R * T)
    v1 = 20.0
    v2 = 0.0

    return SVector(rho, rho * v1, rho * v2, rho * potential_temperature)
end

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerPotentialTemperatureEquations2D()

boundary_conditions = (x_neg = boundary_condition_periodic,
                       x_pos = boundary_condition_periodic,
                       y_neg = boundary_condition_slip_wall,
                       y_pos = boundary_condition_slip_wall)

polydeg = 4
basis = LobattoLegendreBasis(polydeg)

surface_flux = FluxLMARS(340.0)
surface_flux = flux_theta_es
solver = DGSEM(basis, surface_flux)
solver = DGSEM(polydeg = 4 , surface_flux = surface_flux , volume_integral = VolumeIntegralFluxDifferencing(flux_theta))

coordinates_min = (0.0, 0.0)
coordinates_max = (300_000.0, 10_000.0)

cells_per_dimension = (600, 20) # Delta x = Delta z = 1 km
cells_per_dimension = (50, 5)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max,
                      periodicity = (true, false))
initial_condition = initial_condition_gravity_wave

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_gravity,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3000.0)  # 1000 seconds final time

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:entropy_conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     output_directory = "out",
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.85)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation
sol = solve(ode, 
CarpenterKennedy2N54(williamson_condition = false),
#SSPRK43(),
maxiters = 1.0e7,
            dt = 1e-1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks, adaptive = false);

summary_callback()
