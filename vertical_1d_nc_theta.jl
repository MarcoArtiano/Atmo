include("general_setup.jl")
using TrixiAtmo: boundary_condition_slip_wall_2
function prim2velocity(u, x, equations) 
    rho, rho_v1, rho_e, phi = u

    v1 = rho_v1 / rho

    return v1
end

function get_velocity(sol, semi, equations, initial_refinement_level, polydeg)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u_wrap = Trixi.wrap_array(sol.u, semi)
    velocity = Array{Float64}(
        undef,
        polydeg + 1,
        2^initial_refinement_level,
    )
    @unpack node_coordinates = cache.elements
    for element in eachelement(solver, cache)
        for i in eachnode(solver)
            x_local =
                Trixi.get_node_coords(node_coordinates, equations, solver, i, element)
            velocity[i, element] =
                prim2velocity(u_wrap[:, i, element], x_local, equations)
        end
    end

    return velocity

end

function initial_condition_theta(x, t, equations)
    g = 9.81
    c_p = 1004.0
    c_v = 717.0
    gamma = c_p / c_v
    T = 280
    p_0 = 100_000.0  # reference pressure
    R = c_p - c_v    # gas constant (dry air)
    rho0 = p_0/(R*T)
    rho = rho0 * exp(-g/(R*T)*x[1])
    p = rho * (R * T)
    potential_temperature = 300.0
    exner = 1 - g / (c_p * potential_temperature) * x[1]
    # pressure
    p_0 = 100_000.0  # reference pressure
    R = c_p - c_v    # gas constant (dry air)
    p = p_0 * exner^(c_p / R)
    T = potential_temperature * exner

    # density
    rho = p / (R * T)
    v1 = 0.0
    v2 = 0.0
    phi = 9.81*x[1]
    return TrixiAtmo.prim2cons(SVector(rho, v1, p, phi), equations)
end


function vertical_1d_nc(;polydeg::Int, dt::Float64, time_method, initial_refinement_level::Int64, surface_flux, volume_flux, problem_setup::ProblemSetup, use_volume_flux::Bool)
    @unpack equations = problem_setup    

    ode, callbacks, semi  =  setup_problem_vertical_nc(;polydeg = polydeg, dt = dt, initial_refinement_level = initial_refinement_level,
                                   surface_flux = surface_flux, volume_flux = volume_flux,
                                   problem_setup = problem_setup, use_volume_flux = use_volume_flux)

    integrator = init(ode, time_method,
    dt = dt,
    save_everystep = false, callback = callbacks, adaptive = false)
    velocity = NaN
    try
        solve!(integrator)
        velocity = get_velocity(integrator, semi, equations, initial_refinement_level, polydeg)
        return integrator.t, velocity
    catch error
        @info "Blow-up" error
        return integrator.t, velocity
    end

end

function set_vertical_1d_nc(time_method, equations, problem_name)

    return ProblemSetup(problem_name = problem_name, equations = equations, initial_condition = initial_condition_theta, boundary_conditions = boundary_condition_slip_wall_2,
    tspan = (0.0, 48600.0), coordinates_min = (0.0,),
    coordinates_max = (10000.0,), source_terms = nothing, periodicity = false, time_method = time_method)

end


function run_problems(;problem, polydeg, dt, initial_refinement_level, time_method, surface_fluxes, volume_fluxes, use_volume_fluxes)

    @unpack problem_name = problem

    results = DataFrame(surface_flux=String[], volume_flux=String[], use_volume_flux=Bool[], t=Float64[], max_vel=Union{Float64, Nothing}[], min_vel=Union{Float64, Nothing}[])


    for surface_flux in surface_fluxes
        for volume_flux in volume_fluxes
            for use_volume_flux in use_volume_fluxes
                # Setup del problema e salvataggio dei risultati
                t, vel = vertical_1d_nc(;polydeg = polydeg, dt = dt, time_method = time_method, initial_refinement_level = initial_refinement_level,
                                       surface_flux = surface_flux, volume_flux = volume_flux,
                                       problem_setup = problem, use_volume_flux = use_volume_flux)
                # Trova il massimo positivo e il minimo negativo nella matrice vel
                if any(isnan,vel)
                    max_vel = NaN
                    min_vel = NaN
                else
                max_vel = !isempty(vel[vel .> 0]) ? maximum(vel[vel .> 0]) : nothing  # Massimo dei valori positivi
                min_vel = !isempty(vel[vel .< 0]) ? minimum(vel[vel .< 0]) : nothing  # Minimo dei valori negativi
                end
                # Aggiungi i risultati al DataFrame
                push!(results, (string(surface_flux), string(volume_flux), use_volume_flux, t, max_vel, min_vel))
            end
        end
    end

    # Extract the name of the type of time_method
    time_method_name = nameof(typeof(time_method))

    # Constructing the filename for the LaTeX table
    foldername = "ResultsVertical_1D_NC/Refinement_$initial_refinement_level"
    filename = joinpath(foldername, "$(problem_name)_polydeg$(polydeg)_$(time_method_name)_Refinement_$(initial_refinement_level)_dt_$( dt)_gammarho_theta.tex")

    # Open file to write
    open(filename, "w") do file
    # Write the pretty_table output in LaTeX format to the file
    pretty_table(file, results, backend = Val(:latex))
    end

    return nothing

end

function run_vertical_1d_nc(;time_method = RK4(), polydeg::Int = 3)

    Euler1DV = set_vertical_1d_nc(time_method, CompressibleEulerEquations1DNC(1.4), "Euler")

    Potential1DV = set_vertical_1d_nc(time_method, CompressibleEulerPotentialTemperatureEquations1DNC(), "Potential")

    surface_fluxes_euler = (flux_lax_friedrichs, TrixiAtmo.flux_ranocha, TrixiAtmo.flux_hllc, TrixiAtmo.flux_shima_etal, TrixiAtmo.flux_kennedy_gruber, TrixiAtmo.flux_chandrashekar)
    volume_fluxes_euler = (TrixiAtmo.flux_ranocha,TrixiAtmo.flux_shima_etal, TrixiAtmo.flux_kennedy_gruber, TrixiAtmo.flux_chandrashekar)

    surface_fluxes_potential = (flux_lax_friedrichs, flux_theta, flux_theta_rhos, flux_theta_global, flux_theta_es, flux_LMARS )
    volume_fluxes_potential = (flux_theta,  flux_theta_rhos,  flux_theta_global)

    use_volume_fluxes = (true)

    time_methods = (RK4(), SSPRK43())

    polydegs = (3, 4)

    initial_refinement_levels = (3, 4)
    dt = 0.2
        for polydeg in polydegs
            for time_method in time_methods
                for initial_refinement_level in initial_refinement_levels
                run_problems(;problem = Euler1DV, polydeg = polydeg, dt = dt, initial_refinement_level = initial_refinement_level, time_method = time_method, surface_fluxes = surface_fluxes_euler, volume_fluxes = volume_fluxes_euler, use_volume_fluxes = use_volume_fluxes)
                end
            end
        end
	throw(error)
        for polydeg in polydegs
            for time_method in time_methods
                    for initial_refinement_level in initial_refinement_levels
                run_problems(;problem = Potential1DV, polydeg = polydeg, dt = dt, initial_refinement_level = initial_refinement_level, time_method = time_method, surface_fluxes = surface_fluxes_potential, volume_fluxes = volume_fluxes_potential, use_volume_fluxes = use_volume_fluxes)
                    end
            end
        end

    return nothing

end
