using Revise
using Trixi
using OrdinaryDiffEq
using Plots
using TrixiAtmo
using TrixiAtmo: source_terms_convergence_test, initial_condition_convergence_test
cs = 340.0
U = 20.0
N = 0.01

equations_slow = LinearizedGravityWaveEquationsSlow2D(cs, U, N)
equations_fast = LinearizedGravityWaveEquationsFast2D(cs, U, N)
polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lmars)

coordinates_min = (-150_000.0, 0.0)
coordinates_max = (150_000.0, 10_000.0)

trees_per_dimension = (7,4)

 mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
                                      coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                                      periodicity = (true, false), initial_refinement_level = 2)

                                      boundary_conditions = Dict(:y_neg => boundary_condition_slip_wall,
                                      :y_pos => boundary_condition_slip_wall)


initial_condition = initial_condition_convergence_test
semi = SemidiscretizationHyperbolicSplit(
    mesh,
    (equations_slow, equations_fast),
    initial_condition,
    solver,
    solver;
    boundary_conditions = (boundary_conditions, boundary_conditions), 
    source_terms = (source_terms_convergence_test, nothing)
)

function MidPointMatrices(polydeg, mesh, initial_condition, solver)
    equations_vertical = LinearizedGravityWaveEquationsFastVertical2D(cs, U, N)
    semi = SemidiscretizationHyperbolic(mesh, equations_vertical, initial_condition, solver
    ; boundary_conditions = boundary_conditions
    )
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)

    nvariables = (polydeg+1)*(polydeg+1)*length(eachelement(solver,cache))

    jaf = jacobian_ad_forward(semi)

    Mwp = jaf[nvariables + 1:2*nvariables, 2*nvariables+1:3*nvariables]
    Mpw = jaf[2*nvariables+1:3*nvariables, nvariables+1:2*nvariables]

    return Mwp, Mpw
end

Mwp, Mpw = MidPointMatrices(polydeg, mesh, initial_condition, solver)
throw(error)
tspan = (0.0, 3000.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100, solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.1)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks =
    CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = Trixi.solve(
    ode,
    Trixi.SimpleIMEX(;FastMethod = "SE"),
    dt = 0.4, # solve needs some value here but it will be overwritten by the stepsize_callback
    save_everystep = false,
    callback = callbacks,
);