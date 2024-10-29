include("general_setup.jl")

function prim2velocity(u, x, equations) 
    rho, rho_v1, rho_v2, rho_e = u

    v2 = rho_v2 / rho

    return v2
end

function get_velocity(sol, semi, equations, cells_per_dimension, polydeg)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u_wrap = Trixi.wrap_array(sol.u, semi)
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

function initial_condition_gravity_wave(x, t, equations::CompressibleEulerEquations2D)
    

    g = 9.81
    c_p = 1004.0
    c_v = 717.0
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

    # temperature
    T = potential_temperature * exner

    # density
    rho = p / (R * T)

    v1 = 20.0
    v2 = 0.0
    E = c_v * T + 0.5 * (v1^2 + v2^2)
    return SVector(rho, rho * v1, rho * v2, rho * E)
end

function source_terms_gravity(u, x, t, equations)

    rho, _, _, _ = u

    return SVector(zero(eltype(u)), zero(eltype(u)), -9.81*rho, zero(eltype(u)))

end

function inertia(;polydeg::Int, dt::Float64, time_method, cells_per_dimension::Tuple{Int64, Int64}, surface_flux, volume_flux, problem_setup::ProblemSetup, use_volume_flux::Bool)
    @unpack equations = problem_setup    

    ode, callbacks, semi  =  setup_problem_inertia_gravity_wave(;polydeg = polydeg, dt = 0.2, cells_per_dimension = (50, 10),
                                   surface_flux = surface_flux, volume_flux = volume_flux,
                                   problem_setup = problem_setup, use_volume_flux = use_volume_flux)

    integrator = init(ode, time_method,
    dt = dt,
    save_everystep = false, callback = callbacks, adaptive = false)
    velocity = NaN
    x_mesh = 0
    try
        solve!(integrator)
        velocity, x_mesh = get_velocity(integrator, semi, equations, cells_per_dimension, polydeg)
        return integrator.t, velocity, x_mesh
    catch error
        @info "Blow-up" error
        return integrator.t, velocity, x_mesh
    end

end

function set_inertia(time_method, equations, problem_name)

    return ProblemSetup(problem_name = problem_name, equations = equations, initial_condition = initial_condition_gravity_wave,boundary_conditions = (x_neg = boundary_condition_periodic,
    x_pos = boundary_condition_periodic,
    y_neg = boundary_condition_slip_wall,
    y_pos = boundary_condition_slip_wall), tspan = (0.0, 3000.0), coordinates_min = (0.0, 0.0),
    coordinates_max = (300_000.0, 10_000.0), source_terms = source_terms_gravity, periodicity = (true, false), time_method = time_method)

end


function run_problems(;problem, polydeg, time_method, surface_fluxes, volume_fluxes, use_volume_fluxes)

    @unpack problem_name = problem

    results = DataFrame(surface_flux=String[], volume_flux=String[], use_volume_flux=Bool[], t=Float64[], max_positive_vel=Union{Float64, Nothing}[], min_negative_vel=Union{Float64, Nothing}[])


    pl = plot()
    for surface_flux in surface_fluxes
        for volume_flux in volume_fluxes
            for use_volume_flux in use_volume_fluxes
                # Setup del problema e salvataggio dei risultati
                t, vel, x = inertia(;polydeg = polydeg, dt = 0.2, time_method = time_method, cells_per_dimension = (50,10),
                                       surface_flux = surface_flux, volume_flux = volume_flux,
                                       problem_setup = problem, use_volume_flux = use_volume_flux)

                plot!(vec(x),vec(vel), label = string(surface_flux))                                
                # Trova il massimo positivo e il minimo negativo nella matrice vel
                if any(isnan,vel)
                    max_positive_vel = NaN
                    min_negative_vel = NaN
                else
                max_positive_vel = !isempty(vel[vel .> 0]) ? maximum(vel[vel .> 0]) : nothing  # Massimo dei valori positivi
                min_negative_vel = !isempty(vel[vel .< 0]) ? minimum(vel[vel .< 0]) : nothing  # Minimo dei valori negativi
                end
                # Aggiungi i risultati al DataFrame
                push!(results, (string(surface_flux), string(volume_flux), use_volume_flux, t, max_positive_vel, min_negative_vel))
            end
        end
    end

    savefig(pl,"Inertia/$(problem_name)_vol_refined.pdf")
    # Extract the name of the type of time_method
    time_method_name = nameof(typeof(time_method))

    # Constructing the filename for the LaTeX table
    foldername = "INERTIA"
    filename = joinpath(foldername, "$(problem_name)_polydeg$(polydeg)_$(time_method_name)_refined.tex")

    # Open file to write
    open(filename, "w") do file
    # Write the pretty_table output in LaTeX format to the file
    pretty_table(file, results, backend = Val(:latex))
    end

    return nothing

end

function run_inertia(;time_method = RK4(), polydeg::Int = 3)

    Euler1DV = set_inertia(time_method, CompressibleEulerEquations2D(1.4), "Euler")

    Potential1DV = set_inertia(time_method, CompressibleEulerPotentialTemperatureEquations2D(), "Potential")

    surface_fluxes_euler = (flux_lax_friedrichs, FluxLMARS(340.0),)
    volume_fluxes_euler = (flux_ranocha,)

    surface_fluxes_potential = ( FluxLMARS(340.0),)
    volume_fluxes_potential = (flux_theta, flux_theta_global)

    use_volume_fluxes = (true)

    #time_method = RK4()
    time_method = CarpenterKennedy2N54(williamson_condition = false)

    polydeg = 4

                run_problems(;problem = Euler1DV, polydeg = polydeg, time_method = time_method, surface_fluxes = surface_fluxes_euler, volume_fluxes = volume_fluxes_euler, use_volume_fluxes = use_volume_fluxes)



                run_problems(;problem = Potential1DV, polydeg = polydeg, time_method = time_method, surface_fluxes = surface_fluxes_potential, volume_fluxes = volume_fluxes_potential, use_volume_fluxes = use_volume_fluxes)


    return nothing

end   