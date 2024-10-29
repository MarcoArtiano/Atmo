include("general_setup.jl")
using DelimitedFiles
using DataFrames
using CSV
using Plots
using LaTeXStrings
import Trixi: initial_condition_density_wave
function initial_condition_density_wave(x, t, equations::CompressibleEulerPotentialTemperatureEquations1D)
    v1 = 0.1
    rho = 1 + 0.98 * sinpi(2 * (x[1] - t * v1))
    rho_v1 = rho * v1
    p = 20
    rho_theta = (p/equations.p_0)^(1/equations.gamma)*equations.p_0/equations.R
        return SVector(rho, rho_v1, rho_theta)
end

function set_density_wave_1d(time_method, equations, problem_name)

    return ProblemSetup(problem_name = problem_name, equations = equations, initial_condition = initial_condition_density_wave, boundary_conditions = nothing, tspan = (0.0, 40.0), coordinates_min = (-1.0,),
    coordinates_max = (1.0,), source_terms = nothing, periodicity = (true, true), time_method = time_method)
    
end

function run_density_wave_1d(; time_method = SSPRK43(), polydeg::Int = 3)

    EulerTGV = set_density_wave_1d(time_method, CompressibleEulerEquations1D(1.4),"Euler")
    PotentialTGV = set_density_wave_1d(time_method, CompressibleEulerPotentialTemperatureEquations1D(), "Potential")
    polydeg = 3
    initial_refinement_level = 4
    fluxes = (flux_theta, flux_theta_AM, flux_theta_rhos, flux_theta_rhos_AM, flux_theta_global)
    density_wave_1d(;polydeg = polydeg, time_method = time_method, initial_refinement_level = initial_refinement_level, surface_flux = flux_ranocha, volume_flux = flux_ranocha, problem_setup = EulerTGV, use_volume_flux = true)
    for flux in fluxes
    density_wave_1d(;polydeg = polydeg, time_method = time_method, initial_refinement_level = initial_refinement_level, surface_flux = flux, volume_flux = flux, problem_setup = PotentialTGV, use_volume_flux = true)
    end
end

function density_wave_1d(;polydeg::Int, dt = 1.0, time_method, initial_refinement_level::Int64, surface_flux, volume_flux, problem_setup::ProblemSetup, use_volume_flux::Bool)
    @unpack equations, problem_name = problem_setup    

    ode, callbacks, semi  =  setup_problem_density_wave(;polydeg = polydeg, dt = dt, initial_refinement_level = initial_refinement_level,
                                   surface_flux = surface_flux, volume_flux = volume_flux,
                                   problem_setup = problem_setup, use_volume_flux = use_volume_flux)

                                   sol = solve(ode, time_method,
                                   dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                                   save_everystep = false, callback = callbacks, adaptive = false);
    if problem_name == "Euler"
    data = readdlm("DensityWave/analysiseuler.dat", skipstart=1)  
    elseif problem_name == "Potential"
    data = readdlm("DensityWave/analysistheta.dat", skipstart=1)
    end

col_time = 2                 # Time column
col_energy_kinetic = 11       # Energy Kinetic
col_total_energy = 12              # Entropy
col_entropy_phys = 13         # Entropy Phys
col_pressure = 14            # Pressure

time = data[:, col_time]
energy_kinetic = data[:, col_energy_kinetic] .- data[1,col_energy_kinetic]
total_energy = data[:, col_total_energy] .- data[1,col_total_energy]
entropy_phys = data[:, col_entropy_phys] .- data[1,col_entropy_phys]
pressure = data[:, col_pressure] .- data[1,col_pressure]

results = hcat(time, energy_kinetic, total_energy, entropy_phys, pressure)
 
    if problem_name == "Euler"
    writedlm("DensityWave/processed_data_euler.dat", results)    
    elseif problem_name == "Potential"
    writedlm("DensityWave/processed_data_potential_"*string(surface_flux)*".dat", results)    
    end

    return nothing
end

function post_process_density_wave()


    
    # Define the folder containing the data files
    folder_path = "DensityWave"
    
    # Define file names and labels for different datasets
    file_names = [
        "processed_data_euler.dat",
        "processed_data_potential_flux_theta.dat",
        "processed_data_potential_flux_theta_AM.dat",
        "processed_data_potential_flux_theta_global.dat",
        "processed_data_potential_flux_theta_rhos.dat",
       # "processed_data_potential_flux_theta_rhos_AM.dat"
    ]

    #  file_names = [
    #      "processed_data_euler.dat",
    #      "processed_data_potential_flux_theta.dat",
    #      "processed_data_potential_flux_theta_global.dat",
    #      "processed_data_potential_flux_theta_rhos.dat"]
    
    labels = ["Ranocha", "Flux Theta", "Flux Theta AM", "Flux Theta Global", "Flux Theta Rhos", "Flux Theta Rhos AM"]
    labels = ["Ranocha", "Flux Theta", "Flux Theta AM", "Flux Theta Global", "Flux Theta Rhos"]
    #, "Flux Theta Rhos AM"]

     #labels = ["Ranocha", "Flux Theta", "Flux Theta Global", "Flux Theta Rhos"]

    # Read all data into a dictionary of DataFrames
    data_dict = Dict{String, Matrix{Float64}}()
    markers = [:circle, :square, :triangle, :diamond, :star, :cross]
    # Loop through each file and read the data
    for file_name in file_names
        # Construct the full file path
        file_path = joinpath(folder_path, file_name)
        
        # Read the data into a DataFrame
        data_dict[file_name] = readdlm(file_path)  # Salta la riga dell'intestazione

    end
    
    # Get the number of columns (except for the first one, which is time)
    n_columns = size(data_dict[file_names[1]], 2) - 1
    n_columns = 4
    marker_interval = 50
    name = ["Kinetic Energy", "Total Energy", "Entropy", "Pressure"]
    # Generate plots for each column (excluding time)
    for column_index in 2:n_columns + 1  # Start from 2 to include the first column of values
        # Create a new plot for each column
        p = plot()
        
        # Loop through each file to add data to the plot
        for (i, (file_name, label)) in enumerate(zip(file_names, labels))
           
            # Get the time and corresponding column data
            time = data_dict[file_name][:, 1]
            column_data = data_dict[file_name][:, column_index]
            # Plot the current column data
            #plot!(time, column_data, marker=markers[i], label=label)
            plot!(time[1:marker_interval:end], column_data[1:marker_interval:end], 
            marker=markers[i], label=label)
        end
        # Configure the plot
        title!(name[column_index-1])  # Column index - 1 for human-readable numbering
        xlabel!(L"t")
        #ylabel!("Values")
    
        # Save the plot as a PDF
        savename = name[column_index-1]
        pdf_filename = joinpath(folder_path, "DensityWave_"*savename*"complete_ref4_noRHOSAM.pdf")        
        savefig(p, pdf_filename)
    end
    
    println("Plots saved as PDF files.")
    



    return nothing
end