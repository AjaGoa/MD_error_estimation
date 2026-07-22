using Statistics
using Plots
using DelimitedFiles
using Printf
using FFTW 
using Chemfiles
using LsqFit

# --- Simulation Parameters Struct ---
Base.@kwdef struct SimParams
    L_dt::Float64 = 2.0                # in femtoseconds
    s_ene::Int = 100                   # stride
    dt::Float64 = s_ene * L_dt * 1e-3  # picoseconds
    N_atoms::Int = 2048
    molar_mass::Float64 = 39.948
    T_mean_nvt::Float64 = 300.0        # Update this based on actual target or calculated mean
    T_mean_npt::Float64 = 300.0 
end

function block_average(data)
    N = length(data)
    block_sizes = Int[]
    block_errors = Float64[]
    error_of_errors = Float64[]
    
    for B in 1:floor(Int, N/3)
        n_blocks = N ÷ B  # integer division
        blocks = [mean(data[(i-1)*B+1 : i*B]) for i in 1:n_blocks]
        err = std(blocks) / sqrt(n_blocks)
        err_of_err = err / sqrt(2 * (n_blocks - 1))
        
        push!(block_sizes, B)
        push!(block_errors, err)
        push!(error_of_errors, err_of_err)
    end
    return block_sizes, block_errors, error_of_errors
end

error_asymptote_model(x, p) = p[1] .- p[2] .* exp.(-p[3] .* x)

function get_asymptotic_error(sizes::Vector{Int}, errors::Vector{Float64}, N_total::Int)
    n_blocks = N_total ./ sizes
    weights = n_blocks ./ sum(n_blocks) 
    
    max_err = maximum(errors)
    min_err = minimum(errors)
    p0 = [max_err, max_err - min_err, 0.05] 
    
    fit = curve_fit(error_asymptote_model, Float64.(sizes), errors, weights, p0)
    
    p_opt = fit.param
    true_error_estimate = p_opt[1]
    fit_curve = error_asymptote_model(Float64.(sizes), p_opt)
    
    return true_error_estimate, fit_curve
end

function block_average_fp(data::Vector{Float64})
    block_sizes = Int[]
    block_errors = Float64[]
    error_of_errors = Float64[]
    
    current_B = 1
    
    while length(data) >= 2
        n_blocks = length(data)
        
        var_mean = var(data) / n_blocks
        err = sqrt(var_mean)
        err_of_err = err / sqrt(2 * (n_blocks - 1))
        
        push!(block_sizes, current_B)
        push!(block_errors, err)
        push!(error_of_errors, err_of_err)
        
        new_n_blocks = n_blocks ÷ 2
        new_data = zeros(Float64, new_n_blocks)
        for i in 1:new_n_blocks
            new_data[i] = 0.5 * (data[2i - 1] + data[2i])
        end
        
        data = new_data
        current_B *= 2
    end
    
    return block_sizes, block_errors, error_of_errors
end

function identify_plateau(block_sizes, error_block, error_of_errors)
    n = length(error_block)
    
    for i in 1:n
        max_lower = maximum(error_block[j] - error_of_errors[j] for j in i:n)
        min_upper = minimum(error_block[j] + error_of_errors[j] for j in i:n)
        
        if error_block[i] > max_lower && error_block[i] < min_upper
            if i == n
                println("uncertainty estimate did not plateau before the end of the sample.")
            end
            return error_block[i], block_sizes[i], true 
        end
    end
    
    println("No stable global plateau found. Falling back to maximum error.")
    max_idx = argmax(error_block)
    return error_block[max_idx], block_sizes[max_idx], false
end

function autocorrelation(data::Vector{Float64}, mean_shifted=true)
    N = length(data)
    x = mean_shifted ? data .- mean(data) : data
    
    buffer_length = nextpow(2, 2N - 1) 
    buffer_x = zeros(Float64, buffer_length)
    buffer_x[1:N] .= x
    
    F = fft(buffer_x)
    S = abs2.(F) 
    R = real.(ifft(S)) 
    
    acf = R[1:N] 
    lags = 0:(N-1)
    acf ./= (N .- lags)
    acf ./= acf[1] 

    return acf
end

function autocorrelation_loop(data::Vector{Float64}, mean_shifted=true)
    N = length(data)
    μ = mean(data)
    
    x = mean_shifted ? data .- μ : data
    acf = zeros(Float64, N)
    
    for k in 0:(N-1)
        sum_product = 0.0
        for i in 1:(N-k)
            sum_product += x[i] * x[i+k]
        end
        acf[k+1] = sum_product / (N - k)
    end
    
    acf ./= acf[1]
    return acf
end

function run_acf(data::Vector{Float64})
    acf = autocorrelation(data)
    acf_loop = autocorrelation_loop(data)
    return (; acf, acf_loop) 
end

function integrate_acf(acf_array::Vector{Float64})  
    N = length(acf_array)
    tau = 0.0
    
    for t in 1:(N-1)
        # Cut off integration when ACF crosses 0 to prevent integrating noise
        # if acf_array[t+1] <= 0.0
        #     break
        # end
        # Bartlett window weighting (similar to pymbar approaches)
        weight = 1.0 - (t / Float64(N))
        tau += weight * acf_array[t+1]
    end

    return tau
end

function calc_error_acf(tau, N, σ)
    s = max(1.0, 1.0 + 2.0 * tau)  # Statistical inefficiency must be >= 1.0
    return σ * sqrt(s / N)
end

function run_block_averaging(data::Vector{Float64}, N_total::Int)
    block_sizes_fp, error_block_fp, error_of_errors_fp  = block_average_fp(data)
    optimal_error_fp, optimal_size_fp, plateau_found_fp = identify_plateau(block_sizes_fp, error_block_fp, error_of_errors_fp)
    
    block_sizes_blockAvg, block_errors_blockAvg, error_of_errors_blockAvg = block_average(data)
    optimal_error_blockAvg, optimal_size_blockAvg, plateau_found_blockAvg = identify_plateau(block_sizes_blockAvg, block_errors_blockAvg, error_of_errors_blockAvg) 
    true_error_estimate_blockAvg, fit_curve_blockAvg = get_asymptotic_error(block_sizes_blockAvg, block_errors_blockAvg, N_total)
    
    return (; block_sizes_blockAvg, block_errors_blockAvg, true_error_estimate_blockAvg, optimal_error_blockAvg, optimal_size_blockAvg, plateau_found_blockAvg, fit_curve_blockAvg, block_sizes_fp, error_block_fp, error_of_errors_fp, optimal_error_fp, optimal_size_fp, plateau_found_fp)
end

function calculate_cv(E_data::Vector{Float64}, T_mean::Float64, tau::Float64)
    N = length(E_data)
    var_E = var(E_data)
    R = 0.0019872041 # kcal/(mol K)
    Cv = var_E / (R * T_mean^2) 

    # Error in variance: σ_V ≈ V * sqrt(2 * (1 + 2τ) / N)
    s = max(1.0, 1.0 + 2.0 * tau)
    var_E_error = var_E * sqrt(2 * s / N)
    Cv_error = var_E_error / (R * T_mean^2)

    return Cv, Cv_error
end

function calculate_cp(E_pot::Vector{Float64}, E_kin::Vector{Float64}, Press::Vector{Float64}, Dens::Vector{Float64}, T_mean::Float64, tau::Float64, params::SimParams)
    N = length(E_pot)
    N_A = 6.02214076e23 # Avogadro's number
    mass_g = (params.N_atoms * params.molar_mass) / N_A
    
    # V (cm^3) = mass / density -> Convert to A^3 (* 1e24)
    Vol_A3 = (mass_g ./ Dens) .* 1e24
    
    # Enthalpy H = E + PV (Conversion 1 atm * A^3 = 0.0145839 kcal/mol)
    PV_kcal = Press .* Vol_A3 .* 0.0145839
    H = E_pot .+ E_kin .+ PV_kcal
    
    var_H = var(H)
    R = 0.0019872041 # kcal/(mol K)
    Cp = var_H / (R * T_mean^2)
    
    # Error in variance
    s = max(1.0, 1.0 + 2.0 * tau)
    var_H_error = var_H * sqrt(2 * s / N)
    Cp_error = var_H_error / (R * T_mean^2)

    return Cp, Cp_error
end

function compute_stats(data, block::NamedTuple, acf::NamedTuple)
    N = length(data)
    μ = mean(data)
    σ = std(data)
    
    if σ < 1e-10
        return (; μ, σ, naive_error=0.0, error_tau_block_asympt=0.0, error_tau_block_plateau=0.0, error_tau_block_fp_plateau=0.0, error_acf=0.0, error_acf_loop=0.0, acf_abs_diff=zeros(N), acf_max_abs_diff=0.0, tau_acf=0.0) 
    end
    
    naive_error = σ / sqrt(N)

    tau_acf = integrate_acf(acf.acf)
    tau_acf_loop = integrate_acf(acf.acf_loop)

    # Calculate τ equivalents from block methods
    taus = [
        0.5 * ((block.true_error_estimate_blockAvg / naive_error)^2 - 1.0),
        0.5 * ((block.optimal_error_blockAvg / naive_error)^2 - 1.0),
        0.5 * ((block.optimal_error_fp / naive_error)^2 - 1.0),
        tau_acf,
        tau_acf_loop
    ]
    # Bound taus to prevent negative values
    taus = max.(0.0, taus)

    # For-loop mapping for calc_error_acf
    errors = [calc_error_acf(t, N, σ) for t in taus]
    
    acf_abs_diff = abs.(acf.acf .- acf.acf_loop)
    acf_max_abs_diff = maximum(acf_abs_diff)
        
    return (; μ, σ, naive_error, 
              error_tau_block_asympt = errors[1], 
              error_tau_block_plateau = errors[2], 
              error_tau_block_fp_plateau = errors[3], 
              error_acf = errors[4], 
              error_acf_loop = errors[5], 
              acf_abs_diff, acf_max_abs_diff, tau_acf)
end

function compare_methods(data::Vector{Float64}, ensemble_name::String, name::String, stats::NamedTuple, block_avg::NamedTuple, dt::Float64)
    N = length(data)

    if stats.σ < 1e-10
        println(" [$name] Data has ~zero variance, skipping comparison.")
        return nothing
    end

    matrix_width = 182
    println("\n", "="^matrix_width)
    println(" Error-Estimation Method Comparison Matrix: $name ($ensemble_name)")
    println("="^matrix_width)
    @printf(" N = %d, mean = %.6f, std = %.6f\n", N, stats.μ, stats.σ)
    println("-"^matrix_width)
    
    methods = [
        ("Naive (σ/√N)",         stats.naive_error,                "vs Naive"),
        ("Block (fit)",          block_avg.true_error_estimate_blockAvg, "vs Bk(fit)"),
        ("Block (FP plateau)",   block_avg.optimal_error_fp,       "vs FP(plt)"),
        ("Block (plateau)",      block_avg.optimal_error_blockAvg, "vs Bk(plt)"),
        ("Block (fit) τ",        stats.error_tau_block_asympt,     "vs τBk(ft)"),
        ("Block (plateau) τ",    stats.error_tau_block_plateau,    "vs τBk(pl)"),
        ("Block (FP plateau) τ", stats.error_tau_block_fp_plateau, "vs τFP(pl)"),
        ("ACF (FFT)",            stats.error_acf,                  "vs ACF(F)"),
        ("ACF (loop)",           stats.error_acf_loop,             "vs ACF(L)")
    ]
    
    @printf("%-20s | %-12s", "Method", "Error Value")
    for m in methods
        @printf(" | %-10s", m[3])
    end
    println()
    println("-"^matrix_width)
    
    for i in 1:length(methods)
        name_i, err_i, _ = methods[i]
        @printf("%-20s | %-12.6f", name_i, err_i)
        
        for j in 1:length(methods) 
            _, err_j, _ = methods[j]
            if i == j
                @printf(" | %-10s", "    -")
            else
                rel_diff = err_i - err_j 
                @printf(" | %-10.6f", rel_diff)
            end
        end
        println()
    end
    
    println("-"^matrix_width)
    @printf(" Max difference between acf = %.6f\n", stats.acf_max_abs_diff)
    
    return (; 
        naive_error = stats.naive_error, 
        block_fit = block_avg.true_error_estimate_blockAvg, 
        block_fp_plateau = block_avg.optimal_error_fp, 
        block_plateau = block_avg.optimal_error_blockAvg,
        block_tau_fit = stats.error_tau_block_asympt, 
        block_tau_plateau = stats.error_tau_block_plateau,
        block_tau_fp_plateau = stats.error_tau_block_fp_plateau,
        acf_fft = stats.error_acf, 
        acf_loop = stats.error_acf_loop,
        tau_acf = stats.tau_acf
    )
end

function plotting(N, ensemble_name, name, dt, block::NamedTuple, acf::NamedTuple, stats::NamedTuple)
    p1 = plot(block.block_sizes_blockAvg .* dt, block.block_errors_blockAvg, 
            label="simple block err", alpha=0.5, xlabel="Block size (ps)", 
            ylabel="Std. error", title="Block Averaging: simple vs FP", legend=:bottomright)
            
    plot!(p1, block.block_sizes_blockAvg .* dt, block.fit_curve_blockAvg, label="simple fit", linewidth=2, color=:darkorange)
    hline!(p1, [block.true_error_estimate_blockAvg], label="simple asymptote", color=:darkorange, linestyle=:dash)
    
    plot!(p1, block.block_sizes_fp .* dt, block.error_block_fp, label="FP block err", alpha=0.7, color=:purple)
    hline!(p1, [block.optimal_error_fp], label="FP plateau", color=:purple, linestyle=:dash)

    p2 = plot((0:N-1) .* dt, acf.acf, label="ACF (FFT)", linewidth=2, xlabel="Lag (ps)", ylabel="C(t)", title="ACF: FFT vs double loop")
    plot!(p2, (0:N-1) .* dt, acf.acf_loop, label="ACF (loop)", linewidth=1, linestyle=:dot, color=:red)
    hline!(p2, [0], label="", color=:black, alpha=0.4)

    method_labels = ["Naive", "Blk(fit)", "FP(plat)", "Blk(plat)", "Blk(fit)\nτ", "Blk(plat)\nτ", "FP(plat)\nτ", "ACF\n(FFT)", "ACF\n(loop)"]
    method_values = [stats.naive_error, block.true_error_estimate_blockAvg, block.optimal_error_fp, block.optimal_error_blockAvg, stats.error_tau_block_asympt, stats.error_tau_block_plateau, stats.error_tau_block_fp_plateau, stats.error_acf, stats.error_acf_loop]
    p3 = bar(method_labels, method_values, legend=false, ylabel="Estimated std. error", title="Error Estimate Comparison", xrotation=45)
    
    p4 = plot((0:N-1) .* dt, stats.acf_abs_diff, label="Acf-loop and Acf-fft", linewidth=2, xlabel="Lag (ps)", ylabel="diff", title="ACF diff")
    
    combined = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900),
                    plot_title="Method Comparison: $name ($ensemble_name)", margin=8Plots.mm)
    savefig(combined, "comparison_$(ensemble_name)_$(name).svg")
end

function process_ensemble(filename, ensemble_name, col_names, col_indices, params::SimParams)
    println("\n================================================================================")
    println(" Ensemble: $ensemble_name (File: $filename)")
    println("================================================================================")
    
    raw_data = readdlm(filename, skipstart=1)
    N_total = size(raw_data, 1)

    if N_total < 1000
        println(" [WARNING] Data length ($N_total) is very short. Statistics/Errors may be unreliable.")
        println("           Consider extending the simulation time.")
    end

    results = Dict{String, Any}()

    for (name, idx) in zip(col_names, col_indices) 
        println("\n Variable: $name")
        vec_data = convert(Vector{Float64}, raw_data[:, idx])
        N = length(vec_data)

        acf_tuple = run_acf(vec_data)  
        block_tuple = run_block_averaging(vec_data, N)

        stats = compute_stats(vec_data, block_tuple, acf_tuple)
        results[name] = compare_methods(vec_data, ensemble_name, name, stats, block_tuple, params.dt)

        plotting(N, ensemble_name, name, params.dt, block_tuple, acf_tuple, stats)
    end
    
    return results, raw_data
end

function compare_ensembles(nvt_results::Dict, npt_results::Dict)
    common_props = intersect(keys(nvt_results), keys(npt_results))
    
    for prop_name in common_props
        nvt_errors = nvt_results[prop_name]
        npt_errors = npt_results[prop_name]
        
        if nvt_errors === nothing || npt_errors === nothing
            continue
        end

        println("\n", "="^102)
        println(" Cross-Ensemble Error Comparison: $prop_name (NVT vs NpT)")
        println("="^102)
        
        @printf("%-24s | %-14s | %-14s | %-14s | %-14s\n", 
                "Error Method", "NVT Error", "NpT Error", "Difference", "Rel Diff (%)")
        println("-"^102)
        
        # Omit tau_acf from purely error comparisons
        keys_to_compare = filter(k -> k != :tau_acf, keys(nvt_errors))

        for k in keys_to_compare
            err_nvt = getproperty(nvt_errors, k)
            err_npt = getproperty(npt_errors, k)
            
            diff = err_npt - err_nvt
            rel_diff = (err_nvt != 0.0) ? (abs(diff) / abs(err_nvt)) * 100 : 0.0
            
            @printf("%-24s | %-14.6f | %-14.6f | %-14.6f | %-14.2f\n", 
                    string(k), err_nvt, err_npt, diff, rel_diff)
        end
        println("="^102)
    end
end

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

params = SimParams()

# NVT
col_names_nvt = ["Temperature", "Kinetic_Energy", "Potential_Energy", "Conserved_Energy", "Pressure"]
col_indices_nvt = [3, 4, 5, 6, 7]
nvt_results, nvt_raw = process_ensemble("energy_NVT.txt", "NVT", col_names_nvt, col_indices_nvt, params)

# NpT 
col_names_npt = ["Temperature", "Kinetic_Energy", "Potential_Energy", "Conserved_Energy", "Pressure", "Density"]
col_indices_npt = [3, 4, 5, 6, 7, 8]
npt_results, npt_raw = process_ensemble("energy_NpT.txt", "NpT", col_names_npt, col_indices_npt, params)

compare_ensembles(nvt_results, npt_results)

# --- Heat Capacity Calculations ---
println("\n", "="^102)
println(" Heat Capacity Analysis")
println("="^102)

# NVT - Cv Calculation
if haskey(nvt_results, "Kinetic_Energy") && haskey(nvt_results, "Potential_Energy")
    # Total Energy fluctuations for Cv
    E_kin_nvt = convert(Vector{Float64}, nvt_raw[:, 4])
    E_pot_nvt = convert(Vector{Float64}, nvt_raw[:, 5])
    E_tot_nvt = E_kin_nvt .+ E_pot_nvt
    
    # Recalculate tau for Total Energy to ensure rigorous error bounding
    acf_E_tot = run_acf(E_tot_nvt).acf
    tau_E_tot = integrate_acf(acf_E_tot)

    Cv, Cv_err = calculate_cv(E_tot_nvt, params.T_mean_nvt, tau_E_tot)
    @printf(" NVT | C_v = %.6f ± %.6f kcal/(mol K)\n", Cv, Cv_err)
end

# NpT - Cp Calculation
if haskey(npt_results, "Potential_Energy") && haskey(npt_results, "Density")
    E_kin_npt = convert(Vector{Float64}, npt_raw[:, 4])
    E_pot_npt = convert(Vector{Float64}, npt_raw[:, 5])
    Press_npt = convert(Vector{Float64}, npt_raw[:, 7])
    Dens_npt  = convert(Vector{Float64}, npt_raw[:, 8])

    # H = E + PV calculation to find proper tau_H 
    N_A = 6.02214076e23
    mass_g = (params.N_atoms * params.molar_mass) / N_A
    Vol_A3 = (mass_g ./ Dens_npt) .* 1e24
    H_npt = E_pot_npt .+ E_kin_npt .+ (Press_npt .* Vol_A3 .* 0.0145839)

    acf_H = run_acf(H_npt).acf
    tau_H = integrate_acf(acf_H)

    Cp, Cp_err = calculate_cp(E_pot_npt, E_kin_npt, Press_npt, Dens_npt, params.T_mean_npt, tau_H, params)
    @printf(" NpT | C_p = %.6f ± %.6f kcal/(mol K)\n", Cp, Cp_err)
end
println("="^102)