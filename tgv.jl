include("general_setup.jl")
using DelimitedFiles
using DataFrames
using CSV
using Plots
using LaTeXStrings

function initial_condition_taylor_green_vortex(x, t,
	equations::CompressibleEulerPotentialTemperatureEquations3D)
	A = 1.0 # magnitude of speed
	Ms = 0.1 # maximum Mach number

	rho = 1.0
	v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
	v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
	v3 = 0.0
	p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
	p = p +
		1.0 / 16.0 * A^2 * rho *
		(cos(2 * x[1]) * cos(2 * x[3]) + 2 * cos(2 * x[2]) + 2 * cos(2 * x[1]) +
		 cos(2 * x[2]) * cos(2 * x[3]))

	return TrixiAtmo.prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

function set_tgv_3d(time_method, equations, problem_name)

	return ProblemSetup(problem_name = problem_name, equations = equations, initial_condition = initial_condition_taylor_green_vortex, boundary_conditions = nothing, tspan = (0.0, 50.0), coordinates_min = (-1.0, -1.0, -1.0) .* pi,
		coordinates_max = (1.0, 1.0, 1.0) .* pi, source_terms = nothing, periodicity = (true, true, true), time_method = time_method)

end


function initial_condition_taylor_green_vortex(x, t,
	equations::CompressibleEulerEquations3D)
	A = 1.0 # magnitude of speed
	Ms = 0.1 # maximum Mach number

	rho = 1.0
	v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
	v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
	v3 = 0.0
	p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
	p = p +
		1.0 / 16.0 * A^2 * rho *
		(cos(2 * x[1]) * cos(2 * x[3]) + 2 * cos(2 * x[2]) + 2 * cos(2 * x[1]) +
		 cos(2 * x[2]) * cos(2 * x[3]))

	return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

function run_tgv_3d(; time_method = RK4(), polydeg::Int = 3)

	EulerTGV = set_tgv_3d(time_method, CompressibleEulerEquations3D(1.4), "Euler")
	PotentialTGV = set_tgv_3d(time_method, CompressibleEulerPotentialTemperatureEquations3D(), "Potential")
	polydeg = 3
	initial_refinement_level = 2
	fluxes = (flux_theta, flux_theta_AM, flux_theta_rhos, flux_theta_rhos_AM, flux_theta_global)
	tgv_3d(; polydeg = polydeg, time_method = time_method, initial_refinement_level = initial_refinement_level, surface_flux = flux_ranocha, volume_flux = flux_ranocha, problem_setup = EulerTGV, use_volume_flux = true)
	for flux in fluxes
		tgv_3d(; polydeg = polydeg, time_method = time_method, initial_refinement_level = initial_refinement_level, surface_flux = flux, volume_flux = flux, problem_setup = PotentialTGV, use_volume_flux = true)
	end
end

function tgv_3d(; polydeg::Int, dt = 1.0, time_method, initial_refinement_level::Int64, surface_flux, volume_flux, problem_setup::ProblemSetup, use_volume_flux::Bool)
	@unpack equations, problem_name = problem_setup

	ode, callbacks, semi = setup_problem_tgv(; polydeg = polydeg, dt = dt, initial_refinement_level = initial_refinement_level,
		surface_flux = surface_flux, volume_flux = volume_flux,
		problem_setup = problem_setup, use_volume_flux = use_volume_flux)

	sol = solve(ode, time_method,
		dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
		save_everystep = false, callback = callbacks, adaptive = false)
	if problem_name == "Euler"
		data = readdlm("Results/TGV/analysiseuler.dat", skipstart = 1)
	elseif problem_name == "Potential"
		data = readdlm("Results/TGV/analysistheta.dat", skipstart = 1)
	end

	col_time = 2                  # Time column
	col_energy_kinetic = 15       # Energy Kinetic
	col_total_energy = 16         # Entropy
	col_entropy_phys = 17         # Entropy Phys
	col_pressure = 18             # Pressure

	time = data[:, col_time]
	energy_kinetic = data[:, col_energy_kinetic] .- data[1, col_energy_kinetic]
	total_energy = data[:, col_total_energy] .- data[1, col_total_energy]
	entropy_phys = data[:, col_entropy_phys] .- data[1, col_entropy_phys]
	pressure = data[:, col_pressure] .- data[1, col_pressure]

	results = hcat(time, energy_kinetic, total_energy, entropy_phys, pressure)

	if problem_name == "Euler"
		writedlm("Results/TGV/processed_data_euler.dat", results)
	elseif problem_name == "Potential"
		writedlm("Results/TGV/processed_data_potential_" * string(surface_flux) * ".dat", results)
	end

	return nothing
end


function post_process_tgv()

	# Define the folder containing the data files
	folder_path = "Results/TGV"

	# Define file names and labels for different datasets
	file_names = [
		"processed_data_euler.dat",
		"processed_data_potential_flux_theta.dat",
		"processed_data_potential_flux_theta_AM.dat",
		"processed_data_potential_flux_theta_global.dat",
		"processed_data_potential_flux_theta_rhos.dat",
		"processed_data_potential_flux_theta_rhos_AM.dat",
	]

	 file_names = [
	     "processed_data_euler.dat",
	     "processed_data_potential_flux_theta.dat",
	     "processed_data_potential_flux_theta_global.dat",
	     "processed_data_potential_flux_theta_rhos.dat"]

	labels = ["Ranocha", "Flux Theta", "Flux Theta AM", "Flux Theta Global", "Flux Theta Rhos", "Flux Theta Rhos AM"]
	 labels = ["Ranocha", "Flux Theta", "Flux Theta Global", "Flux Theta Rhos"]

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
	marker_interval = 100
	name = ["Kinetic Energy", "Total Energy", "Entropy", "Pressure"]
	# Generate plots for each column (excluding time)
	for column_index in 2:n_columns+1  # Start from 2 to include the first column of values
		# Create a new plot for each column
		p = plot()
        println(column_index)
		# Loop through each file to add data to the plot
		for (i, (file_name, label)) in enumerate(zip(file_names, labels))
            println(file_name)
            println(label)
			# Get the time and corresponding column data
			time = data_dict[file_name][:, 1]
			column_data = data_dict[file_name][:, column_index]
			# Plot the current column data
			#plot!(time, column_data, marker=markers[i], label=label)
			plot!(time[1:marker_interval:end], column_data[1:marker_interval:end],
				marker = markers[i], label = label)
		end
		# Configure the plot
		title!(name[column_index-1])  # Column index - 1 for human-readable numbering
		xlabel!(L"t")
		#ylabel!("Values")
		#println(column_index)
		# Save the plot as a PDF
		savename = name[column_index-1]
        println(savename)
        println(data_dict["processed_data_euler.dat"][:,3])
		pdf_filename = joinpath(folder_path, "TGV_" * savename * "_ref3_2.pdf")
		savefig(p, pdf_filename)
	end

	println("Plots saved as PDF files.")

	return nothing
end
