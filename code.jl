using OrdinaryDiffEq
using Trixi
using TrixiAtmo
using TrixiAtmo: FluxLMARS, source_terms_gravity, flux_theta
using Plots
using LaTeXStrings

polydeg = 3

function prim2thetafunc(u, x, equations::CompressibleEulerPotentialTemperatureEquations2D)
    rho, rho_v1, rho_v2, rho_theta = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    #p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
    p_0 = 100_000.0
    c_p = 1004.0
    c_v = 717.0
    R = c_p - c_v
    gamma = c_p / c_v
    #T = p /(rho*R)
    g = 9.81
    theta = rho_theta / rho - 300.0

    return theta
end

function prim2thetafunceuler(u, x, equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
    p_0 = 100_000.0
    c_p = 1004.0
    c_v = 717.0
    R = c_p - c_v
    gamma = c_p / c_v
    T = p / (rho * R)
    g = 9.81
    theta = T * (p_0 / p)^(R / c_p) - 300.0

    return theta
end

function get_theta(sol, semi, equations, cells_per_dimension)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u_wrap = Trixi.wrap_array(sol.u[end], semi)
    theta = Array{Float64}(
        undef,
        polydeg + 1,
        polydeg + 1,
        cells_per_dimension[1] * cells_per_dimension[2],
    )
    @unpack node_coordinates = cache.elements
    for element in eachelement(solver, cache)
        for j in eachnode(solver), i in eachnode(solver)
            x_local =
                Trixi.get_node_coords(node_coordinates, equations, solver, i, j, element)
            theta[i, j, element] =
                prim2thetafunc(u_wrap[:, i, j, element], x_local, equations)
        end
    end

    return theta

end

function get_thetaeuler(sol, semi, equations, cells_per_dimension)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u_wrap = Trixi.wrap_array(sol.u[end], semi)
    theta = Array{Float64}(
        undef,
        polydeg + 1,
        polydeg + 1,
        cells_per_dimension[1] * cells_per_dimension[2],
    )
    @unpack node_coordinates = cache.elements
    for element in eachelement(solver, cache)
        for j in eachnode(solver), i in eachnode(solver)
            x_local =
                Trixi.get_node_coords(node_coordinates, equations, solver, i, j, element)
            theta[i, j, element] =
                prim2thetafunceuler(u_wrap[:, i, j, element], x_local, equations)
        end
    end

    return theta

end

function initial_condition_warm_bubble(
    x,
    t,
    equations::CompressibleEulerPotentialTemperatureEquations2D,
)
    g = equations.g
    c_p = equations.c_p
    c_v = equations.c_v
    # center of perturbation
    center_x = 10000.0
    center_z = 2000.0
    # radius of perturbation
    radius = 2000.0
    # distance of current x to center of perturbation
    r = sqrt((x[1] - center_x)^2 + (x[2] - center_z)^2)

    # perturbation in potential temperature
    potential_temperature_ref = 300.0
    potential_temperature_perturbation = 0.0
    if r <= radius
        potential_temperature_perturbation = 2 * cospi(0.5 * r / radius)^2
    end
    potential_temperature = potential_temperature_ref + potential_temperature_perturbation

    # Exner pressure, solves hydrostatic equation for x[2]
    exner = 1 - g / (c_p * potential_temperature_ref) * x[2]

    # pressure
    p_0 = 100_000.0  # reference pressure
    R = c_p - c_v    # gas constant (dry air)
    p = p_0 * exner^(c_p / R)
    T = potential_temperature * exner

    # density
    rho = p / (R * T)
    v1 = 20.0
    v2 = 0.0
    phi = equations.g * x[2] * 0
    rho_theta = (p / equations.p_0)^(1 / equations.gamma) * equations.p_0 / equations.R

    return SVector(rho, rho * v1, rho * v2, rho * potential_temperature)
end

###############################################################################
# semidiscretization of the compressible Euler equations
struct WarmBubbleSetup
    # Physical constants
    g::Float64       # gravity of earth
    c_p::Float64     # heat capacity for constant pressure (dry air)
    c_v::Float64     # heat capacity for constant volume (dry air)
    gamma::Float64   # heat capacity ratio (dry air)

    function WarmBubbleSetup(; g = 9.81, c_p = 1004.0, c_v = 717.0, gamma = c_p / c_v)
        new(g, c_p, c_v, gamma)
    end
end

function cons2theta(u, equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
    p_0 = 100_000.0
    c_p = 1004.0
    c_v = 717.0
    R = c_p - c_v
    gamma = c_p / c_v
    rho_theta = (p / p_0)^(1 / gamma) * p_0 / R

    return SVector(rho, v1, v2, rho_theta)
end

function Trixi.varnames(::typeof(cons2theta), ::CompressibleEulerEquations2D)
    ("rho", "rho_v1", "rho_v2", "rho_theta")
end

# Initial condition
function (setup::WarmBubbleSetup)(x, t, equations::CompressibleEulerEquations2D)
    @unpack g, c_p, c_v = setup

    # center of perturbation
    center_x = 10000.0
    center_z = 2000.0
    # radius of perturbation
    radius = 2000.0
    # distance of current x to center of perturbation
    r = sqrt((x[1] - center_x)^2 + (x[2] - center_z)^2)

    # perturbation in potential temperature
    potential_temperature_ref = 300.0
    potential_temperature_perturbation = 0.0
    if r <= radius
        potential_temperature_perturbation = 2 * cospi(0.5 * r / radius)^2
    end
    potential_temperature = potential_temperature_ref + potential_temperature_perturbation

    # Exner pressure, solves hydrostatic equation for x[2]
    exner = 1 - g / (c_p * potential_temperature) * x[2]

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



# Source terms
@inline function (setup::WarmBubbleSetup)(u, x, t, equations::CompressibleEulerEquations2D)
    @unpack g = setup
    rho, _, rho_v2, _ = u
    return SVector(zero(eltype(u)), zero(eltype(u)), -g * rho, -g * rho_v2)
end

###############################################################################
# semidiscretization of the compressible Euler equations
function solverEuler(Nx, Ny, form)
    warm_bubble_setup = WarmBubbleSetup()

    equations = CompressibleEulerEquations2D(warm_bubble_setup.gamma)

    boundary_conditions = (
        x_neg = boundary_condition_periodic,
        x_pos = boundary_condition_periodic,
        y_neg = boundary_condition_slip_wall,
        y_pos = boundary_condition_slip_wall,
    )

    polydeg = 3
    basis = LobattoLegendreBasis(polydeg)

    # This is a good estimate for the speed of sound in this example.
    # Other values between 300 and 400 should work as well.
    surface_flux = FluxLMARS(340.0)
    #surface_flux = flux_shima_etal
    #volume_flux = flux_kennedy_gruber
    volume_flux = flux_shima_etal
    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

    if form == "vol"
        solver = DGSEM(basis, surface_flux, volume_integral)
    elseif form == "weak"
        solver = DGSEM(basis, surface_flux)
    end

    coordinates_min = (0.0, 0.0)
    coordinates_max = (20_000.0, 10_000.0)

    cells_per_dimension = (Nx, Ny)
    mesh = StructuredMesh(
        cells_per_dimension,
        coordinates_min,
        coordinates_max,
        periodicity = (true, false),
    )

    semi = SemidiscretizationHyperbolic(
        mesh,
        equations,
        warm_bubble_setup,
        solver,
        source_terms = warm_bubble_setup,
        boundary_conditions = boundary_conditions,
    )

    ###############################################################################
    # ODE solvers, callbacks etc.

    tspan = (0.0, 1000.0)  # 1000 seconds final time

    ode = semidiscretize(semi, tspan)

    summary_callback = SummaryCallback()

    analysis_interval = 1000

    analysis_callback = AnalysisCallback(
        semi,
        interval = analysis_interval,
        extra_analysis_errors = (:entropy_conservation_error,),
    )

    alive_callback = AliveCallback(analysis_interval = analysis_interval)

    save_solution = SaveSolutionCallback(
        interval = analysis_interval,
        save_initial_solution = true,
        save_final_solution = true,
        output_directory = "out",
        solution_variables = cons2prim,
    )

    stepsize_callback = StepsizeCallback(cfl = 1.0)

    callbacks = CallbackSet(
        summary_callback,
        analysis_callback,
        alive_callback,
        save_solution,
        stepsize_callback,
    )

    ###############################################################################
    # run the simulation
    sol = solve(
        ode,
        CarpenterKennedy2N54(williamson_condition = false),
        maxiters = 1.0e7,
        dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
        save_everystep = false,
        callback = callbacks,
    )

    summary_callback()

    theta = get_thetaeuler(sol, semi, equations, cells_per_dimension)

    return theta, sol, semi

end

function solverTheta(Nx, Ny, form)

    equations = CompressibleEulerPotentialTemperatureEquations2D()

    boundary_conditions = (
        x_neg = boundary_condition_periodic,
        x_pos = boundary_condition_periodic,
        y_neg = boundary_condition_slip_wall,
        y_pos = boundary_condition_slip_wall,
    )

    polydeg = 3
    basis = LobattoLegendreBasis(polydeg)

    # This is a good estimate for the speed of sound in this example.
    # Other values between 300 and 400 should work as well.
    surface_flux = FluxLMARS(340.0)
    surface_flux = flux_theta_global_es

    volume_flux = flux_theta
    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)


    if form == "weak"
        solver = DGSEM(basis, surface_flux)
    elseif form == "vol"
        solver = DGSEM(basis, surface_flux, volume_integral)
    end
    coordinates_min = (0.0, 0.0)
    coordinates_max = (20_000.0, 10_000.0)

    cells_per_dimension = (Nx, Ny)
    mesh = StructuredMesh(
        cells_per_dimension,
        coordinates_min,
        coordinates_max,
        periodicity = (true, false),
    )
    initial_condition = initial_condition_warm_bubble

    semi = SemidiscretizationHyperbolic(
        mesh,
        equations,
        initial_condition,
        solver,
        source_terms = source_terms_gravity,
        boundary_conditions = boundary_conditions,
    )

    ###############################################################################
    # ODE solvers, callbacks etc.

    tspan = (0.0, 1000.0)  # 1000 seconds final time

    ode = semidiscretize(semi, tspan)

    summary_callback = SummaryCallback()

    analysis_interval = 1000

    analysis_callback = AnalysisCallback(
        semi,
        interval = analysis_interval,
        extra_analysis_errors = (:entropy_conservation_error,),
    )

    alive_callback = AliveCallback(analysis_interval = analysis_interval)

    save_solution = SaveSolutionCallback(
        interval = analysis_interval,
        save_initial_solution = true,
        save_final_solution = true,
        output_directory = "out",
        solution_variables = cons2prim,
    )

    stepsize_callback = StepsizeCallback(cfl = 1.0)

    callbacks = CallbackSet(
        summary_callback,
        analysis_callback,
        alive_callback,
        save_solution,
        stepsize_callback,
    )

    ###############################################################################
    # run the simulation
    sol = solve(
        ode,
        #RK4(),
        CarpenterKennedy2N54(williamson_condition = false),
        maxiters = 1.0e7,
        dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
        save_everystep = false,
        callback = callbacks,
    )

    summary_callback()

    theta = get_theta(sol, semi, equations, cells_per_dimension)

    return theta, sol, semi

end


function main()
    # Nx = 32
    # Ny = 16
    # form = "weak"
    # PlotTheta(Nx, Ny, form)
    # form = "vol"
    # PlotTheta(Nx, Ny, form)

    Nx = 64
    Ny = 32
    form = "weak"
    minE1, maxE1, minT1, maxT1 = PlotTheta(Nx, Ny, form)
    form = "vol"
    minE2, maxE2, minT2, maxT2 = PlotTheta(Nx, Ny, form)
    println(minE1)
    println(maxE1)
    println(minT1)
    println(maxT1)
    println(minE2)
    println(maxE2)
    println(minT2)
    println(maxT2)
    # Nx = 128
    # Ny = 64
    # form = "weak"
    # PlotTheta(Nx, Ny, form)
    # form = "vol"
    # PlotTheta(Nx, Ny, form)

    # Nx = 130
    # Ny = 66
    # form = "weak"
    # PlotTheta(Nx, Ny, form)
    # form = "vol"
    # PlotTheta(Nx, Ny, form)

    #Nx = 256
    #Ny = 128
    #form = "weak"
    #PlotTheta(Nx,Ny,form)
    #form = "vol"
    #PlotTheta(Nx,Ny,form)

    return nothing
end

function PlotTheta(Nx, Ny, form)

    thetaT, solT, semiT = solverTheta(Nx, Ny, form)
    thetaE, solE, semiE = solverEuler(Nx, Ny, form)
    climmax = maximum((maximum(thetaE), maximum(thetaT)))
    climmin = minimum((minimum(thetaE), minimum(thetaT)))

    fontsizes() = (xtickfontsize = 18,
               ytickfontsize = 18,
               legendfontsize = 18,
               xguidefontsize = 20,
               yguidefontsize = 20,
               titlefontsize = 20,)

    pT = plot(
        ScalarPlotData2D(
            thetaT,
            semiT))

            figT = plot(pT;
            title = "Potential temperature perturbation [K]",
            clims = (-0.6, 2.5),
            size = (1200, 600),
            aspect_ratio = :equal,
            margin = 0.6 * Plots.cm,
            xticks = ([0, 5_000, 10_000, 15_000, 20_000],
                      ["0", "5", "10", "15", "20"]),
            xguide = L"$x$ [km]",
            yticks = ([0, 2_500, 5_000, 7_500, 10_000],
                      ["0", "2.5", "5.0", "7.5", "10.0"]),
            yguide = L"$z$ [km]",
            fontsizes()...)        

    pE = plot(
        ScalarPlotData2D(
            thetaE,
            semiE))

            figE = plot(pE;
            title = "Potential temperature perturbation [K]",
            clims = (-0.6, 2.5),
            size = (1200, 600),
            aspect_ratio = :equal,
            margin = 0.6 * Plots.cm,
            xticks = ([0, 5_000, 10_000, 15_000, 20_000],
                      ["0", "5", "10", "15", "20"]),
            xguide = L"$x$ [km]",
            yticks = ([0, 2_500, 5_000, 7_500, 10_000],
                      ["0", "2.5", "5.0", "7.5", "10.0"]),
            yguide = L"$z$ [km]",
            fontsizes()...)    
            
    nameend = "_N" * string(Nx) * "x" * string(Ny) * "_CFL_1_testing_ES_diviso340.pdf"
    titlePotential = form * "Potential" * nameend
    titleEuler = form * "Euler" * nameend
    savefig(figT, titlePotential)
    savefig(figE, titleEuler)

    return minimum(thetaE), maximum(thetaE), minimum(thetaT), maximum(thetaT)
end


#subspotential!(sol,semi,equations)
