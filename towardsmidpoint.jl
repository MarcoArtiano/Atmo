using Revise
using Trixi
using OrdinaryDiffEq
using Plots
using TrixiAtmo
using TrixiAtmo: LinearizedGravityWaveEquationsFastVertical2D
using TrixiAtmo: source_terms_convergence_test, initial_condition_convergence_test
cs = 340.0
U = 20.0
N = 0.01
equations_vertical = LinearizedGravityWaveEquationsFastVertical2D(cs, U, N)
equations_slow = LinearizedGravityWaveEquationsSlow2D(cs, U, N)
equations_fast = LinearizedGravityWaveEquationsFast2D(cs, U, N)
polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lmars)
equations_vertical = LinearizedGravityWaveEquationsFastVertical2D(cs, U, N)

coordinates_min = (-150_000.0, 0.0)
coordinates_max = (150_000.0, 10_000.0)

trees_per_dimension = (7,4)

 mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
                                      coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                                      periodicity = (true, false), initial_refinement_level = 2)

                                      boundary_conditions = Dict(:y_neg => boundary_condition_slip_wall,
                                      :y_pos => boundary_condition_slip_wall)




function initial_conditions_debugging(x,t, equations)
    r = sqrt(x[1]^2+x[2]^2)
    return SVector(1*r,2*r*0,3*r*0,4*r*0)
end

initial_condition = initial_condition_convergence_test
#initial_condition = initial_conditions_debugging

semi = SemidiscretizationHyperbolicSplit(
    mesh,
    (equations_slow, equations_fast),
    initial_condition,
    solver,
    solver;
    boundary_conditions = (boundary_conditions, boundary_conditions), 
    source_terms = (source_terms_convergence_test, nothing)
)

function MidPointMatrices(polydeg, mesh, initial_condition, solver, boundary_conditions)
    equations_vertical = LinearizedGravityWaveEquationsFastVertical2D(cs, U, N)
    equations2 = LinearizedGravityWaveEquationsFast2D(cs,U,N)
    semi = SemidiscretizationHyperbolic(mesh, equations_vertical, initial_condition, solver
    ; boundary_conditions = boundary_conditions)
    mesh, _, solver, cache = Trixi.mesh_equations_solver_cache(semi)

    nvariables = (polydeg+1)*(polydeg+1)*length(eachelement(solver,cache))

    jaf = jacobian_ad_forward(semi)
    Mwp = jaf[2:4:4*nvariables,3:4:4*nvariables]
    Mpw = jaf[3:4:4*nvariables,2:4:4*nvariables]

    semi = SemidiscretizationHyperbolic(mesh, equations2, initial_condition, solver
    ; boundary_conditions = boundary_conditions)
    mesh, _, solver, cache = Trixi.mesh_equations_solver_cache(semi)


    jaf = jacobian_ad_forward(semi)
    Mwp = jaf[2:4:4*nvariables,3:4:4*nvariables]
    Mwu = jaf[2:4:4*nvariables,1:4:4*nvariables]
    Mww = jaf[2:4:4*nvariables,2:4:4*nvariables]
    Mwb = jaf[2:4:4*nvariables,4:4:4*nvariables]
    Mpw = jaf[3:4:4*nvariables,2:4:4*nvariables]

    return Mwp, Mpw, Mwu, Mww, Mwb
end

function DefineOperators_hacky(polydeg, mesh, initial_condition, solver, boundary_conditions)
    equations_vertical = LinearizedGravityWaveEquationsFastVertical2D(cs, U, N)
    semi = SemidiscretizationHyperbolic(mesh, equations_vertical, initial_condition, solver
    ; boundary_conditions = boundary_conditions)
    @unpack cache, boundary_conditions = semi
    nvariables = (polydeg+1)*(polydeg+1)*length(eachelement(solver,cache))
    u = zeros(nvariables*4)
    jaf = zeros(nvariables*4,nvariables*4)
    u = zeros(nvariables*4)
    println(boundary_conditions)
    du = zeros(nvariables*4)
    for i in 1:nvariables*4
        u = zeros(nvariables*4)
        du = zeros(nvariables*4)
        u[i] = 1.0
        u_wrap = Trixi.wrap_array(u, semi)
        du_wrap = Trixi.wrap_array(du, semi)
        Trixi.rhs!(du_wrap, u_wrap, 2.0, mesh, equations_vertical, boundary_conditions, nothing, solver, cache)
        jaf[:,i] .= du
    end
    Mwp = jaf[2:4:4*nvariables,3:4:4*nvariables]
    Mpw = jaf[3:4:4*nvariables,2:4:4*nvariables]
    return Mwp, Mpw
end

function plot_eigenvalues(matrix)
    # Calcola gli autovalori della matrice
    eigenvalues = eigvals(matrix)

    # Estrai la parte reale e immaginaria degli autovalori
    real_parts = real(eigenvalues)
    imag_parts = imag(eigenvalues)

    # Crea il grafico scatter degli autovalori
    scatter(real_parts, imag_parts,
            xlabel = "Re", ylabel = "Im",
            legend = false, title = "Eigenvalues on Complex Plane",
            marker = (:circle, 8, :blue),
            xlims = (-maximum(abs, real_parts) - 0.05, maximum(abs, real_parts) + 0.05),
            ylims = (-maximum(abs, imag_parts) - 0.05, maximum(abs, imag_parts) + 0.05))

    # Aggiungi linee tratteggiate per gli assi Re e Im
    plot!([-maximum(abs, real_parts) - 0.05, maximum(abs, real_parts) + 0.05], [0, 0], 
          line = (:dash, :black))  # Linea orizzontale (asse x)
    plot!([0, 0], [-maximum(abs, imag_parts) - 0.05, maximum(abs, imag_parts) + 0.05], 
          line = (:dash, :black))  # Linea verticale (asse y)

          println(maximum(real_parts))
          println(maximum(imag_parts))
end

dt = 0.4
Mwp, Mpw, Mwu, Mww, Mwb = MidPointMatrices(polydeg, mesh, initial_condition, solver, boundary_conditions)
throw(error)
#Mwp3, Mpw3 = DefineOperators_hacky(polydeg, mesh, initial_condition, solver, boundary_conditions)
T = 500.0
tspan = (0.0, T)
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
FastMethod = "RK4"
# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = Trixi.solve(
    ode,
    Trixi.SimpleIMEX(;FastMethod = FastMethod, Mwp, Mpw, Mww, equations_vertical),
    dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
    save_everystep = false,
    callback = callbacks,
);

pd = PlotData2D(sol)
a = plot(pd["p"], aspect_ratio = 10)
savefig(a,"test_p_$(FastMethod)_$T.pdf")

a = plot(pd["u"], aspect_ratio = 10)
savefig(a,"test_u_$(FastMethod)_$T.pdf")

a = plot(pd["w"], aspect_ratio = 10)
savefig(a,"test_w_$(FastMethod)_$T.pdf")

a = plot(pd["b"], aspect_ratio = 10)
savefig(a,"test_b_$(FastMethod)_$T.pdf")