using Statistics
using Plots
using DelimitedFiles
using Printf
using FFTW 
using Chemfiles

# https://juliamath.github.io/FFTW.jl/stable/fft/
# Tuckerman - L 13.4

# Ruzne pristupy k blokum ? jak poznat spravnou velikost a minimalni mnozstvi bloku ? 
function block_average(data)
    N = length(data)
    block_sizes = Int[]
    block_errors = Float64[]
    
    for B in 1:floor(Int, N/10)
        n_blocks = N ÷ B 
        if n_blocks < 5
            break 
        end
        blocks = [mean(data[(i-1)*B+1 : i*B]) for i in 1:n_blocks] # Better then maximum ? 
        push!(block_sizes, B)
        push!(block_errors, std(blocks) / sqrt(n_blocks))
    end
    return block_sizes, block_errors
end

function autocorrelation(data::Vector{Float64}, max_lag::Int)
    N = length(data)
    x = data .- mean(data)
    
    # fft runs faster when array length is power of 2, 2N - 1 is minimum length 
    buffer_length = nextpow(2, 2N - 1) # nextpow(a, x) - The smallest a^n not less than x, where n is a non-negative integer. a must be greater than 1, and x must be greater than 0
    buffer_x = zeros(Float64, buffer_length)
    buffer_x[1:N] .= x
    
    F = fft(buffer_x)
    S = abs2.(F) 
    R = real.(ifft(S)) # still need real. after abs2 because of the floating point errors in IFFT
    
    acf = R[1:N] # < - je to R ofsetnute ? ma to byt od 0 nebo 1:N? 
    lags = 0:(N-1)
    acf ./= (N .- lags) # N - k normalization 
    acf ./= acf[1] # normalize to C(0) = 1.0 
    # zkontrolovat normalizaci - je nulovy posun na indexu 0 nebo 1 ? prohlidnout si nejdriv jak vypada cela acf, a pak az ji normalizovat a rezat
    # fftfreq - neco uvidim :)

    return acf[1:max_lag+1] # max_lag is N/5 - after this acf is too noisy - so not include the rest - and + 1 is because of the 0 lag (to get the acf(max_lag) instead of acf(max_lag-1))
end

function compute_stats(data)
    N = length(data)
    μ = mean(data)
    σ = std(data)
    
    if σ < 1e-10
        return μ, σ, 0.0, Int[], Float64[], 0.0, Float64[], 0, 0.0, 1.0, 0.0 
    end
    
    naive_error = σ / sqrt(N)
    sizes, errors_block = block_average(data)
    
    if isempty(errors_block)
        est_err_block = 0.0
        println("empty ?")
    else
        tail_length = max(1, floor(Int, length(errors_block) * 0.1)) # calculated last 10% of the tot number of block error point, floor - round down to int, max(1, .. ) - to get at least 1
        est_err_block = mean(errors_block[end-tail_length+1:end]) # averaging last 10% of the block error points
        # nebylo by lepsi vzit prumer par maxim treba po 50%? 
    end
    
    max_lag = floor(Int, N/5)
    acf = autocorrelation(data, max_lag)
    
    cutoff_idx = findfirst(x -> x <= 0.0, acf) # kde spravna by se mela uriznout? 
    if isnothing(cutoff_idx) 
        cutoff_idx = length(acf)
    end
    
    τ = sum(acf[2:cutoff_idx]) 
    s = max(1.0, 1 + 2 * τ) # to not get - value or 0
    error_acf = σ * sqrt(s / N)
    
    return μ, σ, naive_error, sizes, errors_block, est_err_block, acf, cutoff_idx, τ, s, error_acf
end

function format_val(val, is_constant)
    return is_constant ? "Constant" : @sprintf("%.6f", val)
end

# Tuckerman 4.5 
function calculate_cv(variance_PE::Float64, T_mean::Float64)
    R = 0.0019872041 # gas constant kcal/(mol K)
    Cv_excess = variance_PE / (R * T_mean^2) # tady by mela byt celkova, mozna celkova teplota ensamplu misto T_mean? 
    #Cv_ideal = 1.5 * R # Monoatomic 3D ideal gas contribution
    #Cv_total = Cv_excess + Cv_ideal
    
    return Cv_excess # , Cv_total
end

function calculate_cp(E_pot::Vector{Float64}, E_kin::Vector{Float64}, Press::Vector{Float64}, Dens::Vector{Float64}, T_mean::Float64; N_atoms=2048, molar_mass=39.948)
    N_A = 6.02214076e23 # Avogadro's number
    mass_g = (N_atoms * molar_mass) / N_A
    
    # V (cm^3) = mass / density -> Convert to A^3 (* 1e24)
    Vol_A3 = (mass_g ./ Dens) .* 1e24
    
    # Enthalpy H = E + PV (Conversion 1 atm * A^3 = 0.0145839 kcal/mol)
    PV_kcal = Press .* Vol_A3 .* 0.0145839
    H = E_pot .+ E_kin .+ PV_kcal
    
    R = 0.0019872041 # kcal/(mol K)
    variance_H = var(H)
    Cp_total = variance_H / (R * T_mean^2)
    
    return Cp_total
end

function process_ensemble(filename, ensemble_name, col_names, col_indices)
    println(" Ensemble: $ensemble_name (File: $filename)")
    
    raw_data = readdlm(filename, skipstart=1)

    temp_idx = findfirst(x -> x == "Temperature", col_names)
    T_mean = isnothing(temp_idx) ? println("Temperature column not found") : mean(convert(Vector{Float64}, raw_data[:, col_indices[temp_idx]]))

    for (name, idx) in zip(col_names, col_indices) # poparuje jmeno promenne a index sloupce v souboru
        println(" Variable: $name")
        
        vec_data = convert(Vector{Float64}, raw_data[:, idx])
        stats = compute_stats(vec_data)
        is_const = stats[2] < 1e-10 
        
        @printf("%-25s | %-20s\n", "Metric", "Value") 
        @printf("--------------------------|----------------------\n")
        @printf("%-25s | %-20.6f\n", "Mean", stats[1])
        @printf("%-25s | %-20.6f\n", "Std Dev", stats[2])
        @printf("%-25s | %-20s\n", "Naive Error", format_val(stats[3], is_const))
        @printf("--------------------------|----------------------\n")
        @printf("%-25s | %-20s\n", "Est. Error (Block Avg)", format_val(stats[6], is_const))
        @printf("%-25s | %-20s\n", "Est. Error (ACF)", format_val(stats[11], is_const))
        @printf("--------------------------|----------------------\n")
        @printf("%-25s | %-20s\n", "Autocorr. Time (τ)", format_val(stats[9], is_const))
        @printf("%-25s | %-20s\n", "Stat Inefficiency (s)", format_val(stats[10], is_const))
        
        # Tuckerman 4.5
        # https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/University_Physics_II_-_Thermodynamics_Electricity_and_Magnetism_(OpenStax)/03%3A_The_First_Law_of_Thermodynamics/3.06%3A_Heat_Capacities_of_an_Ideal_Gas
        # C_v
        if name == "Potential_Energy" && ensemble_name == "NVT"
            # Δ E = \sqrt (Cpk_b T^2)
            variance_PE = stats[2]^2
            Cv_excess = calculate_cv(variance_PE, T_mean)  / 2.048 # per atom and convert to cal/mol K, because E pot is sum over the whole box
            # Cv_total ? 
            println("\n Heat Capacity (Cv) per atom:")
            @printf("  Cv (excess) = %.6f cal/(mol K)\n", Cv_excess)
            #@printf("  Cv (total)  = %.6f kcal/(mol K)\n", Cv_total)
        end
        
        # C_p
        if name == "Density" && ensemble_name == "NpT"
            idx_pe   = col_indices[findfirst(x -> x == "Potential_Energy", col_names)]
            idx_ke   = col_indices[findfirst(x -> x == "Kinetic_Energy", col_names)]
            idx_p    = col_indices[findfirst(x -> x == "Pressure", col_names)]
            idx_dens = col_indices[findfirst(x -> x == "Density", col_names)]
            
            E_pot = convert(Vector{Float64}, raw_data[:, idx_pe])
            E_kin = convert(Vector{Float64}, raw_data[:, idx_ke])
            Press = convert(Vector{Float64}, raw_data[:, idx_p])
            Dens  = convert(Vector{Float64}, raw_data[:, idx_dens])
            
            Cp_total = calculate_cp(E_pot, E_kin, Press, Dens, T_mean) / 2.048
            println("\n Isobaric Heat Capacity (Cp):")
            @printf("  Cp (total)  = %.6f cal/(mol K)\n", Cp_total)
        end

        if !is_const
            sizes, errors_block = stats[4], stats[5]
            acf, cutoff_idx = stats[7], stats[8]
            naive_err = stats[3]
            
            p1 = plot(sizes, errors_block, title="Block Averaging", xlabel="Block Size (B)", ylabel="Standard Error", label="Block Error", linewidth=2, legend=:bottomright)
            hline!(p1, [naive_err], label="Naive Error", linestyle=:dash, color=:red)

            p2 = plot(0:cutoff_idx-1, acf[1:cutoff_idx], title="Autocorrelation", xlabel="Lag (steps)", ylabel="C(t)", label="ACF", linewidth=2)
            hline!(p2, [0], label="", color=:black, alpha=0.5)

            combined = plot(p1, p2, layout=(1, 2), size=(900, 400), plot_title="Analysis: $name ($ensemble_name)", margin=5Plots.mm)
            
            plot_filename = "analysis_$(ensemble_name)_$(name).png"
            savefig(combined, plot_filename)
        end
    end
end

# NVT
col_names_nvt = ["Temperature", "Kinetic_Energy", "Potential_Energy", "Conserved_Energy", "Pressure"]
col_indices_nvt = [3, 4, 5, 6, 7]
process_ensemble("energy_NVT.txt", "NVT", col_names_nvt, col_indices_nvt)

# NpT 
col_names_npt = ["Temperature", "Kinetic_Energy", "Potential_Energy", "Conserved_Energy", "Pressure", "Density"]
col_indices_npt = [3, 4, 5, 6, 7, 8]
process_ensemble("energy_NpT.txt", "NpT", col_names_npt, col_indices_npt)
