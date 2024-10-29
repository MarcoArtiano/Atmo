using OrdinaryDiffEq
using Trixi
using TrixiAtmo
using Plots
using DataFrames
using PrettyTables

Base.@kwdef struct ProblemSetup{NameT, EquationsT, InitialConditionT, BoundaryConditionT, TspanT, CoordinateT, SourceT, PeriodicityT, TimeMethodT}
    problem_name::NameT
    equations::EquationsT
    initial_condition::InitialConditionT
    boundary_conditions::BoundaryConditionT
    tspan::TspanT
    coordinates_min::CoordinateT
    coordinates_max::CoordinateT
    source_terms::SourceT
    periodicity::PeriodicityT
    time_method::TimeMethodT
end

function setup_problem(;polydeg::Int, dt::Float64, cells_per_dimension::Tuple{Int64}, surface_flux, volume_flux, problem_setup::ProblemSetup, use_volume_flux::Bool)

    @unpack tspan, periodicity, coordinates_min, coordinates_max, equations = problem_setup                         
    @unpack initial_condition, source_terms, boundary_conditions, time_method = problem_setup
        
    if use_volume_flux
        solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux, volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
    else
        solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)
    end

    mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max, periodicity = periodicity)
    
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms = source_terms,
    boundary_conditions = boundary_conditions)

    ode = semidiscretize(semi, tspan)
    
    summary_callback = SummaryCallback()
    
    analysis_interval = 10000
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
    
    alive_callback = AliveCallback(analysis_interval = analysis_interval)

    save_solution = SaveSolutionCallback(interval = 10000,
                                         save_initial_solution = true,
                                         save_final_solution = true,
                                         solution_variables = cons2prim)
    
    callbacks = CallbackSet(summary_callback,
                            analysis_callback,
                            alive_callback,
                            save_solution)

    return ode, callbacks, semi                   
    
end

function setup_problem_tgv(;polydeg::Int, dt::Float64, initial_refinement_level::Int64, surface_flux, volume_flux, problem_setup::ProblemSetup, use_volume_flux::Bool)

    @unpack tspan, periodicity, coordinates_min, coordinates_max, equations = problem_setup                         
    @unpack initial_condition, source_terms, boundary_conditions, time_method = problem_setup
    @unpack problem_name = problem_setup
    if use_volume_flux
        solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux, volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
    else
        solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)
    end

    mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level = initial_refinement_level,
    n_cells_max = 100_000)    
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

    ode = semidiscretize(semi, tspan)
    
    summary_callback = SummaryCallback()
    if problem_name == "Euler"
    analysis_interval = 100
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval, save_analysis = true,output_directory="TGV",analysis_filename="analysiseuler.dat",extra_analysis_integrals=(energy_kinetic,energy_total, entropy, pressure))
    elseif problem_name == "Potential"
    analysis_interval = 100
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval, save_analysis = true,output_directory="TGV",analysis_filename="analysistheta.dat",extra_analysis_integrals=(TrixiAtmo.energy_kinetic,TrixiAtmo.entropy,TrixiAtmo.entropy_phys,TrixiAtmo.pressurecompute, ))
    end
    alive_callback = AliveCallback(analysis_interval = analysis_interval)

    save_solution = SaveSolutionCallback(interval = 100,
                                         save_initial_solution = true,
                                         save_final_solution = true,
                                         solution_variables = cons2prim)
    
    stepsize_callback = StepsizeCallback(cfl = 0.01)

    callbacks = CallbackSet(summary_callback,
                            analysis_callback,
                            alive_callback,
                            save_solution,
                            stepsize_callback)

    return ode, callbacks, semi                       
    
end

function setup_problem_density_wave(;polydeg::Int, dt::Float64, initial_refinement_level::Int64, surface_flux, volume_flux, problem_setup::ProblemSetup, use_volume_flux::Bool)

    @unpack tspan, periodicity, coordinates_min, coordinates_max, equations = problem_setup                         
    @unpack initial_condition, source_terms, boundary_conditions, time_method = problem_setup
    @unpack problem_name = problem_setup
    if use_volume_flux
        solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux, volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
    else
        solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)
    end

    mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level = initial_refinement_level,
    n_cells_max = 100_000)    
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

    ode = semidiscretize(semi, tspan)
    
    summary_callback = SummaryCallback()
    if problem_name == "Euler"
    analysis_interval = 100
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval, save_analysis = true,output_directory="DensityWave",analysis_filename="analysiseuler.dat",extra_analysis_integrals=(energy_kinetic,energy_total, entropy, pressure))
    elseif problem_name == "Potential"
    analysis_interval = 100
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval, save_analysis = true,output_directory="DensityWave",analysis_filename="analysistheta.dat",extra_analysis_integrals=(TrixiAtmo.energy_kinetic,TrixiAtmo.entropy,TrixiAtmo.entropy_phys,TrixiAtmo.pressurecompute, ))
    end
    alive_callback = AliveCallback(analysis_interval = analysis_interval)

    save_solution = SaveSolutionCallback(interval = 100,
                                         save_initial_solution = true,
                                         save_final_solution = true,
                                         solution_variables = cons2prim)
    
    stepsize_callback = StepsizeCallback(cfl = 0.05)

    callbacks = CallbackSet(summary_callback,
                            analysis_callback,
                            alive_callback,
                            save_solution,
                            stepsize_callback)

    return ode, callbacks, semi                       
    
end

function setup_problem_vertical_nc(;polydeg::Int, dt::Float64, initial_refinement_level::Int64, surface_flux, volume_flux, problem_setup::ProblemSetup, use_volume_flux::Bool)

    @unpack tspan, periodicity, coordinates_min, coordinates_max, equations = problem_setup                         
    @unpack initial_condition, source_terms, boundary_conditions, time_method = problem_setup
        
    if use_volume_flux
        solver = DGSEM(polydeg = polydeg, surface_flux = (surface_flux, flux_nonconservative_gravity), volume_integral = VolumeIntegralFluxDifferencing((volume_flux, flux_nonconservative_gravity)))
    else
        solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)
    end

    mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 60_000, periodicity = periodicity)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms = source_terms,
    boundary_conditions = boundary_conditions)

    ode = semidiscretize(semi, tspan)
    
    summary_callback = SummaryCallback()
    
    analysis_interval = 10000
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
    
    alive_callback = AliveCallback(analysis_interval = analysis_interval)

    save_solution = SaveSolutionCallback(interval = 10000,
                                         save_initial_solution = true,
                                         save_final_solution = true,
                                         solution_variables = cons2prim)
    
    callbacks = CallbackSet(summary_callback,
                            analysis_callback,
                            alive_callback,
                            save_solution)

    return ode, callbacks, semi                   
    
end

function setup_problem_inertia_gravity_wave(;polydeg::Int, dt::Float64, cells_per_dimension::Tuple{Int64, Int64}, surface_flux, volume_flux, problem_setup::ProblemSetup, use_volume_flux::Bool)

    @unpack tspan, periodicity, coordinates_min, coordinates_max, equations = problem_setup                         
    @unpack initial_condition, source_terms, boundary_conditions, time_method = problem_setup
        
    if use_volume_flux
        solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux, volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
    else
        solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)
    end

    mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max, periodicity = periodicity)
    
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms = source_terms,
    boundary_conditions = boundary_conditions)

    ode = semidiscretize(semi, tspan)
    
    summary_callback = SummaryCallback()
    
    analysis_interval = 10000
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
    
    alive_callback = AliveCallback(analysis_interval = analysis_interval)

    save_solution = SaveSolutionCallback(interval = 10000,
                                         save_initial_solution = true,
                                         save_final_solution = true,
                                         solution_variables = cons2prim)
    
                                         stepsize_callback = StepsizeCallback(cfl = 0.85)

    callbacks = CallbackSet(summary_callback,
                            analysis_callback,
                            alive_callback,
                            save_solution,
                            stepsize_callback)

    return ode, callbacks, semi                   
    
end

function (flux_lmars::FluxLMARS)(u_ll, u_rr, orientation::Integer,
    equations::CompressibleEulerEquations1D)
c = flux_lmars.speed_of_sound

# Unpack left and right state
rho_ll, v1_ll, p_ll = cons2prim(u_ll, equations)
rho_rr, v1_rr, p_rr = cons2prim(u_rr, equations)


v_ll = v1_ll
v_rr = v1_rr
rho = 0.5 * (rho_ll + rho_rr)
p = 0.5 * (p_ll + p_rr) - 0.5 * c * rho * (v_rr - v_ll)
v = 0.5 * (v_ll + v_rr) - 1 / (2 * c * rho) * (p_rr - p_ll)

# We treat the energy term analogous to the potential temperature term in the paper by
# Chen et al., i.e. we use p_ll and p_rr, and not p
if v >= 0
f1, f2, f3 = v * u_ll
f3 = f3 + p_ll * v
else
f1, f2, f3 = v * u_rr
f3 = f3 + p_rr * v
end

f2 = f2 + p


return SVector(f1, f2, f3)
end
