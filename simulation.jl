# Error estimation in molecular dynamics
# Perform analysis of statistical uncertainty of averages obtained from molecular dynamics simulations. Run molecular dynamics of a simple liquid system and pick several scalar quantities of
# interest, such as potential, kinetic, or total energy, the distance between two atoms (within one
# molecule) or an angle, for example. First, calculate the average and the standard deviation of
# these quantities. Then, consider the statistical uncertainty of the average given that the data is
# correlated in time. Use the method of block averages to estimate the uncertainty of the mean.
# Further, calculate the autocorrelation function and from it the autocorrelation time to estimate
# the statistical inefficiency of the molecular dynamics sampling. Based on that, estimate the
# uncertainty of the mean again and compare the result to that obtained from block averages.
# • Variant A: Use the liquid Argon system and compare NVT and NPT results.

# https://github.com/songbin6280/MD_NVT_NH_Python/blob/master/serial_d.py
# https://github.com/JuliaMolSim/Molly.jl/blob/master/src/simulators.jl

using StaticArrays
using Random
using Statistics
using Printf

mutable struct MDParameters
    N_atoms::Int
    rho::Float64
    T_target::Float64
    P_target::Float64
    dt::Float64
    rcut::Float64
    n_steps::Int
    eq_steps::Int
    tau_T::Float64
    tau_P::Float64
    beta_T::Float64
end

function init_parameters()
    return MDParameters(
        256,      # N_atoms (argon near triple point)
        0.8442,   # rho
        0.728,    # T_target
        0.05,     # P_target (reduced units)
        0.005,    # dt (Time step)
        2.5,      # rcut (Cutoff radius)
        10000,    # n_steps (Production steps)
        2000,     # eq_steps (Equilibration steps)
        0.1,      # tau_T (Temperature coupling time)
        1.0,      # tau_P (Pressure coupling time)
        0.1       # beta_T (Isothermal compressibility estimate)
    )
end

function init_positions(N, L)
    n_c = ceil(Int, N^(1/3))  # round up to the nearest integer for cubic lattice
    spacing = L / n_c
    pos = SVector{3, Float64}[] # SVector - StaticArrays for fixed-size vectors, more memory efficient

    for x in 0:n_c-1, y in 0:n_c-1, z in 0:n_c-1
        if length(pos) < N
            push!(pos, SVector((x+0.5)*spacing, (y+0.5)*spacing, (z+0.5)*spacing))
        end
    end
    return pos
end

function init_velocities(N, T)
    vel = [SVector{3, Float64}(randn(), randn(), randn()) for _ in 1:N]
    
    v_cm = sum(vel) ./ N 
    vel = [v .- v_cm for v in vel] # subtract center-of-mass velocity to ensure zero net momentum
    
    current_T = sum(sum(v.^2) for v in vel) / (3 * N)  # from equipartition theorem KE = (3/2) * N * k_B * T, and KE = (1/2) * m * sum(v^2), so T = (2/3N) * KE in reduced units
    scale_factor = sqrt(T / current_T) 
    return [v .* scale_factor for v in vel]
end

function compute_forces_pe_virial(pos, L, rcut)
    N = length(pos)
    forces = zeros(SVector{3, Float64}, N)
    pe = 0.0
    virial = 0.0
    rcut2 = rcut^2
    inv_L = 1.0 / L
    
    for i in 1:N-1
        for j in i+1:N
            # Minimum image convention, [-L/2, L/2], get shortest distance
            rij = pos[i] .- pos[j]
            rij = rij .- L .* round.(rij .* inv_L)
            r2 = sum(rij.^2)
            
            if r2 < rcut2
                r6inv = (1.0 / r2)^3
                r12inv = r6inv^2
                
                # LJ Potential
                pe += 4.0 * (r12inv - r6inv)
                
                # virial contribution for pressure calculation ( r_ij . f_ij )
                virial_ij = 24.0 * (2.0 * r12inv - r6inv)

                virial += virial_ij

                # Force magnitude / r
                fij = virial_ij / r2
                f_vec = fij .* rij
                
                forces[i] += f_vec
                forces[j] -= f_vec 
            end
        end
    end
    return forces, pe, virial
end

# floor() - rounds down to nearest int
#ceil() - rounds up to nearest int
# round() - rounds to nearest int 

function run_simulation(ensemble::Symbol, params::MDParameters)
    N_atoms = params.N_atoms
    dt = params.dt
    rcut = params.rcut
    T_target = params.T_target
    tau_P = params.tau_P
    tau_T = params.tau_T
    P_target = params.P_target

    # box lengt h 
    L = (N_atoms / params.rho)^(1/3)

    pos = init_positions(N_atoms, L)
    vel = init_velocities(N_atoms, T_target)

    forces, current_pe, virial = compute_forces_pe_virial(pos, L, rcut)

    current_T = (sum(sum(v.^2) for v in vel) * 2.0) / (3.0 * N_atoms)
    current_V = L^3
    current_P = (N_atoms * current_T + virial / 3.0) / current_V

    for step in 1:params.eq_steps
        # velocity verlet
        vel .= vel .+ 0.5 .* forces .* dt # half step vel 
        
        pos .= pos .+ vel .* dt # full step pos
        pos .= [p .- L .* floor.(p ./ L) for p in pos]
        
        forces, current_pe, virial = compute_forces_pe_virial(pos, L, rcut)
        
        vel .= vel .+ 0.5 .* forces .* dt
        
        # Velocity Rescaling - only every 10th step to avoid over-constraining the dynamics but also controlling temperature 
        if step % 10 == 0
            curr_T = sum(sum(v.^2) for v in vel) / (3 * N_atoms)
            vel .= vel .* sqrt(T_target / curr_T)
        end
    end

    # Nosé-Hoover
    
    zeta = 0.0  # Thermostat friction
    eta = 0.0   # Barostat strain rate
    W_barostat = 3.0 * N_atoms * T_target * tau_P^2 # Barostat mass (controls how fast the volume responds)

    filename = "argon_data_$(ensemble).txt"
    open(filename, "w") do io
        println(io, "Step  PE  KE  Total_E  Dist_1_2  Temp  Press  Volume")
        
        for step in 1:params.n_steps
            #first half-step thermostat/barostat 
            zeta += (dt / 2.0) * (current_T / T_target - 1.0) / tau_T^2
            if ensemble == :NPT
                eta += (dt / 2.0) * current_V * (current_P - P_target) / W_barostat
            end
            
            total_friction = ensemble == :NPT ? (zeta + eta) : zeta
            vel .= vel .* exp(-total_friction * (dt / 2.0))

            # first half-step velocity 
            vel .= vel .+ 0.5 .* forces .* dt

            # full step pos 
            if ensemble == :NPT
                scale_factor = exp(eta * dt)
                L *= scale_factor
                pos .= pos .* scale_factor
            end
            
            pos .= pos .+ vel .* dt
            pos .= [p .- L .* floor.(p ./ L) for p in pos] # Minimum image

            forces, current_pe, virial = compute_forces_pe_virial(pos, L, rcut)

            # second half-step velocity 
            vel .= vel .+ 0.5 .* forces .* dt

            # second half-step thermostat/barostat
            # Re-calculate intermediate properties for the second friction update
            current_ke = 0.5 * sum(sum(v.^2) for v in vel)
            current_T = (2.0 * current_ke) / (3.0 * N_atoms)
            current_V = L^3
            current_P = (N_atoms * current_T + virial / 3.0) / current_V

            vel .= vel .* exp(-total_friction * (dt / 2.0))
            
            zeta += (dt / 2.0) * (current_T / T_target - 1.0) / tau_T^2
            if ensemble == :NPT
                eta += (dt / 2.0) * current_V * (current_P - P_target) / W_barostat
            end

            current_te = current_pe + current_ke
            
            # distance between 2 atoms (minimum image) 
            r12 = pos[1] .- pos[2]
            r12 = r12 .- L .* round.(r12 ./ L)
            dist_12 = sqrt(sum(r12.^2))

            @printf(io, "%d  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f\n", 
                    step, current_pe, current_ke, current_te, dist_12, current_T, current_P, current_V)
        end
    end
end

params = init_parameters()
# @time
run_simulation(:NVT, params)
run_simulation(:NPT, params)
