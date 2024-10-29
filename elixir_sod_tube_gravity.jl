using OrdinaryDiffEq
using Trixi

function initial_condition_sod_shock_tube_gravity(x, t, equations::CompressibleEulerEquations1D)
    # domain must be set to [0, 1], γ = 2, final time = 0.12
    rho = x[1] < 0.5 ? 1.0 : 0.125
    v1 = 0.0
    p = x[1] < 0.5 ? 1.0 : 0.1
    return prim2cons(SVector(rho, v1, p), equations)
end

@inline function boundary_condition_transmissive(u_inner,
    orientation::Integer, direction, x,
    t,
    surface_flux_function,
    equations::CompressibleEulerEquations1D)
    u_boundary = initial_condition_sod_shock_tube_gravity(x, t, equations)
    surface_flux_function = flux_lax_friedrichs
return surface_flux_function(u_inner, u_boundary, orientation, equations)
end

function source_terms_gravity(u, x, t, equations)

    rho, _, _ = u

    return SVector(zero(eltype(u)), -rho*1, zero(eltype(u)))

end

equations = CompressibleEulerEquations1D(1.4)
initial_condition = initial_condition_sod_shock_tube_gravity
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

surface_flux = flux_lax_friedrichs
volume_flux = flux_chandrashekar
basis = LobattoLegendreBasis(4)
shock_indicator_variable = density_pressure
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = shock_indicator_variable)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)
#solver = DGSEM(basis,surface_flux)
coordinates_min = (0.0,)
coordinates_max = (1.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 10_000, periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions = boundary_conditions
, source_terms = source_terms_gravity)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)



stepsize_callback = StepsizeCallback(cfl = 0.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks, adaptive = false);
summary_callback() # print the timer summary
