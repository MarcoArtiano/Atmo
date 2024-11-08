using Trixi
using TrixiAtmo
using SummationByPartsOperators
using OrdinaryDiffEq
polydeg = -1
flux_splitting = TrixiAtmo.splitting_vanleer_haenel
accuracy_order = 6
initial_refinement_level = 2
source_of_coefficients = Mattsson2017
nnodess = 16
equations = CompressibleEulerPotentialTemperatureEquations2D()
tol = 1e-6
function initial_condition(x, t, equations::CompressibleEulerPotentialTemperatureEquations2D)
    # change discontinuity to tanh
    # typical resolution 128^2, 256^2
    # domain size is [-1,+1]^2
    slope = 15
    B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
    rho = 0.5 + 0.75 * B
    v1 = 0.5 * (B - 1)
    v2 = 0.1 * sin(2 * pi * x[1])
    p = 1.0
    return TrixiAtmo.prim2cons(SVector(rho, v1, v2, p), equations)
end

if polydeg === -1
    # Use upwind SBP discretization
    D_upw = upwind_operators(source_of_coefficients;
                             derivative_order = 1,
                             accuracy_order,
                             xmin = -1.0, xmax = 1.0,
                             N = nnodess)
    solver = FDSBP(D_upw,
                   surface_integral = SurfaceIntegralUpwind(flux_splitting),
                   volume_integral = VolumeIntegralUpwind(flux_splitting))

    @info "Kelvin-Helmholtz instability" accuracy_order nnodes initial_refinement_level flux_splitting
else
    # Use DGSEM
    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
    solver = DGSEM(; polydeg, surface_flux = FluxLMARS(340.0), volume_integral)

    @info "Kelvin-Helmholtz instability" polydeg initial_refinement_level volume_flux
end

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max;
                initial_refinement_level,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
@show Trixi.ndofs(semi)

tspan = (0.0, 15.0)
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

stepsize_callback = StepsizeCallback(cfl = 0.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution)
                        
                        # sol = solve(ode, SSPRK43(); controller = PIDController(0.55, -0.27, 0.05),
                        # abstol = tol, reltol = tol,
                        # ode_default_options()...)
                        # pd = PlotData2D(sol)
                        # plot(pd["rho"])

                        integrator = init(ode, SSPRK43(); controller = PIDController(0.55, -0.27, 0.05),
                        abstol = tol, reltol = tol,
                        ode_default_options()...)
  
      try
          solve!(integrator)
      catch error
          @info "Blow-up" integrator.t
          reset_threads!()
      end

#    sol = solve(ode, SSPRK43(), maxiters = 1.0e7,# solve needs some value here but it will be overwritten by the stepsize_callback
#    save_everystep = false, callback = callbacks; controller = PIDController(0.55, -0.27, 0.05),
#                      abstol = tol, reltol = tol,
#                      ode_default_options()...)

#                    sol = solve(ode, RK4(), maxiters = 1.0e7,
#  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#  save_everystep = false, callback = callbacks)