using Statistics
using Plots
using DelimitedFiles
using Printf
using FFTW 
using Chemfiles
using LsqFit

# https://juliamath.github.io/FFTW.jl/stable/fft/
# Tuckerman - L 13.4
# Error estimates on averages of correlated data - H. Flyvbjerg and H. G. Petersen
# FP .py  : https://github.com/rsdefever/block_average/blob/master/block_average/block_average.py
# https://github.com/choderalab/pymbar/blob/main/pymbar/timeseries.py
# Allen, M. P., & Tildesley, D. J. — Computer Simulation of Liquids - Ch 6.4 - statistical inefficiency

function block_average(data)
    N = length(data)
    block_sizes = Int[]
    block_errors = Float64[]
    
    for B in 1:floor(Int, N/3)
        n_blocks = N ÷ B 
        blocks = [mean(data[(i-1)*B+1 : i*B]) for i in 1:n_blocks]
        push!(block_sizes, B)
        push!(block_errors, std(blocks) / sqrt(n_blocks))
    end
    return block_sizes, block_errors
end

# p[1] = asymptote (true error), p[2] = c (scaling factor),  p[3] = k (decay rate)
error_asymptote_model(x, p) = p[1] .- p[2] .* exp.(-p[3] .* x)

function get_asymptotic_error(sizes::Vector{Int}, errors::Vector{Float64}, N_total::Int)
    # weight - More blocks = less noise = higher weight in the fit.
    n_blocks = N_total ./ sizes
    weights = n_blocks ./ sum(n_blocks) 
    
    # Initial parameter guess: [Asymptote, Scale, Rate]
    max_err = maximum(errors)
    min_err = minimum(errors)
    p0 = [max_err, max_err - min_err, 0.05] 
    
    fit = curve_fit(error_asymptote_model, Float64.(sizes), errors, weights, p0)
    
    p_opt = fit.param
    true_error_estimate = p_opt[1]
    
    fit_curve = error_asymptote_model(Float64.(sizes), p_opt)
    
    return true_error_estimate, fit_curve
end

# Flyvbjerg and Petersen - 1989 - Error estimates on averages of correlated data
function block_average_fp(data::Vector{Float64})
    block_sizes = Int[]
    block_errors = Float64[]
    error_of_errors = Float64[]
    
    current_B = 1
    
    # Iterate by halving the dataset size until fewer than 2 blocks remain
    while length(data) >= 2
        n_blocks = length(data)
        
        # Standard error of the mean for the current block size
        var_mean = var(data) / n_blocks
        err = sqrt(var_mean)
        
        # Flyvbjerg-Petersen Error of the Error
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

# inspo: https://github.com/rsdefever/block_average/blob/master/block_average/block_average.py
function identify_plateau(block_sizes, error_block, error_of_errors)
    n = length(error_block)
    
    for i in 1:n
        # check difference with highest and lowest error of errors value
        max_lower = maximum(error_block[j] - error_of_errors[j] for j in i:n)
        min_upper = minimum(error_block[j] + error_of_errors[j] for j in i:n)
        
        # Check if the current point intersects all remaining error bars
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
    
    # fft runs faster when array length is power of 2, 2N - 1 is minimum length 
    buffer_length = nextpow(2, 2N - 1) 
    buffer_x = zeros(Float64, buffer_length)
    buffer_x[1:N] .= x
    
    F = fft(buffer_x)
    S = abs2.(F) 
    R = real.(ifft(S)) 
    
    acf = R[1:N] 
    lags = 0:(N-1)
    acf ./= (N .- lags) # N - k normalization 
    acf ./= acf[1]      # normalize to C(0) = 1.0 

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

function calculate_cv(variance_E::Float64, T_mean::Float64)
    R = 0.0019872041 # gas constant kcal/(mol K)
    Cv = variance_E / (R * T_mean^2) 
    return Cv
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

function run_block_averaging(data::Vector{Float64}, N_total::Int)
    block_sizes_fp, error_block_fp, error_of_errors_fp  = block_average_fp(data)
    optimal_error_fp, optimal_size_fp, plateau_found_fp = identify_plateau(block_sizes_fp, error_block_fp, error_of_errors_fp)
    
    block_sizes_blockAvg, block_errors_blockAvg = block_average(data)
    true_error_estimate_blockAvg, fit_curve_blockAvg = get_asymptotic_error(block_sizes_blockAvg, block_errors_blockAvg, N_total)
    
    return (; block_sizes_blockAvg, block_errors_blockAvg, true_error_estimate_blockAvg, fit_curve_blockAvg, block_sizes_fp, error_block_fp, error_of_errors_fp, optimal_error_fp, optimal_size_fp, plateau_found_fp)
end

function run_acf(data::Vector{Float64})
    acf = autocorrelation(data)
    acf_loop = autocorrelation_loop(data)

    return (; acf, acf_loop) 
end

function calc_error_acf(tau, N, σ)
    s = max(1.0, 1 + 2 * tau)  # sem by mela prijit ( 1 - t/N) prakticky jako vaha ? <-- pymbar
    return σ * sqrt(s / N)
end

function compute_stats(data, block::NamedTuple, acf::NamedTuple)
    N = length(data)
    μ = mean(data)
    σ = std(data)
    
    if σ < 1e-10
        return (; μ, σ, naive_error=0.0, true_err_block=0.0, tau_block=0.0, error_acf=0.0, tau_acf=0.0, s_acf=1.0) 
    end
    
    # Naive statistics
    naive_error = σ / sqrt(N)

    # ACF
    tau_acf = sum(acf.acf[2:N]) # mozna ukoncit driv nez to hitne ten noisy konec ? 
    error_acf = calc_error_acf(tau_acf, N, σ)

    tau_acf_loop = sum(acf.acf_loop[2:N]) 
    error_acf_loop = calc_error_acf(tau_acf_loop, N, σ)
    
    acf_abs_diff = abs.(acf.acf .- acf.acf_loop)
    acf_max_abs_diff = maximum( acf_abs_diff)
    
    # Block
    tau_block = 0.5 * ((block.true_error_estimate_blockAvg / naive_error)^2 - 1.0) # TODO ozdrojovat - Extract: $\tau = 0.5 \left( \left( \frac{\epsilon_{block}}{\epsilon_{naive}} \right)^2 - 1 \right)$Recalculate: $\epsilon_{recalc} = \epsilon_{naive} \sqrt{1 + 2\tau}$
    error_tau_block = calc_error_acf(tau_block, N, σ)
        
    tau_block_fp = 0.5 * ((block.optimal_error_fp / naive_error)^2 - 1.0)
    error_tau_block_fp = calc_error_acf(tau_block_fp, N, σ)

    # nesla by tady udelat for loop pres tau aby spocitala error ? 
        
    return (; μ, σ, naive_error, error_tau_block, error_tau_block_fp, error_acf, error_acf_loop, acf_abs_diff, acf_max_abs_diff) # true_err_block = true_err_from_fit, 

end

function compare_methods(data::Vector{Float64}, ensemble_name::String, name::String, stats::NamedTuple, block_avg::NamedTuple, dt::Float64)
    N = length(data)

    if stats.σ < 1e-10
        println(" [$name] Data has ~zero variance, skipping comparison.")
        return nothing
    end

    println("\n", "="^102)
    println(" Error-Estimation Method Comparison Matrix: $name ($ensemble_name)")
    println("="^102)
    @printf(" N = %d, mean = %.6f, std = %.6f\n", N, stats.μ, stats.σ)
    println("-"^102)
    
    methods = [
        ("Naive (σ/√N)", stats.naive_error),
        ("Block (fit)",  block_avg.true_error_estimate_blockAvg),
        ("Block (FP)",   block_avg.optimal_error_fp),
        ("Block (fit) from τ", stats.error_tau_block),
        ("Block (FP) from τ", stats.error_tau_block_fp),
        ("ACF (FFT)",    stats.error_acf),
        ("ACF (loop)",   stats.error_acf_loop)
    ]
    
    # Print the Matrix Header
    @printf("%-18s | %-12s | %-10s | %-10s | %-10s | %-10s | %-10s\n", 
            "Method", "Error Value", "vs Naive", "vs Blk(fit)", "vs Blk(FP)", "vs ACF(FFT)", "vs ACF(Lp)")
    println("-"^102)
    
    for i in 1:length(methods)
        name_i, err_i = methods[i]
        
        # 1. Print the row name and the actual error value
        @printf("%-18s | %-12.6f", name_i, err_i)
        
        # 2. Loop through again to calculate differences for the columns
        for j in 1:length(methods)
            name_j, err_j = methods[j]
            
            if i == j
                # Diagonal of the matrix (comparing a method to itself)
                @printf(" | %-10s", "    -")
            else
                rel_diff = err_i - err_j 
                @printf(" | %-12.6f", rel_diff)
            end
        end

        println("="^102)
    end
    
    println("-"^102)
    @printf(" Max difference between acf = %.6f\n", stats.acf_max_abs_diff)
    
end

function process_ensemble(filename, ensemble_name, col_names, col_indices, dt)
    println(" Ensemble: $ensemble_name (File: $filename)")

    raw_data = readdlm(filename, skipstart=1)

    T_mean = 0.0
    vec_E_kin = Float64[]
    vec_E_pot = Float64[]
    vec_Press = Float64[]
    vec_Dens = Float64[]
    comparison_results = Dict{String, Any}()

    for (name, idx) in zip(col_names, col_indices) 
        println("\n Variable: $name")
        vec_data = convert(Vector{Float64}, raw_data[:, idx])
        N = length(vec_data)

        acf_tuple = run_acf(vec_data)  
        block_tuple = run_block_averaging(vec_data, N)

        stats = compute_stats(vec_data, block_tuple, acf_tuple)
        compare_methods(vec_data, ensemble_name, name, stats, block_tuple, dt)

        plotting(N, ensemble_name, name, dt, block_tuple, acf_tuple, stats)
    end
end

function plotting(N, ensemble_name, name, dt, block::NamedTuple, acf::NamedTuple, stats::NamedTuple)
       
    p1 = plot(block.block_sizes_blockAvg .* dt, block.block_errors_blockAvg, 
            label="simple block err", 
            alpha=0.5,
            xlabel="Block size (ps)", 
            ylabel="Std. error",
            title="Block Averaging: simple vs FP", 
            legend=:bottomright)
    plot!(p1, block.block_sizes_blockAvg .* dt, block.fit_curve_blockAvg, label="simple fit", linewidth=2, color=:darkorange)
    hline!(p1, [block.true_error_estimate_blockAvg], label="simple asymptote", color=:darkorange, linestyle=:dash)
    plot!(p1, block.block_sizes_fp .* dt, block.error_block_fp, label="FP block err", alpha=0.7, color=:purple)
    hline!(p1, [block.optimal_error_fp], label="FP plateau", color=:purple, linestyle=:dash)

    p2 = plot((0:N-1) .* dt, acf.acf, label="ACF (FFT)", linewidth=2, xlabel="Lag (ps)", ylabel="C(t)", title="ACF: FFT vs double loop")
    plot!(p2, (0:N-1) .* dt, acf.acf_loop, label="ACF (loop)", linewidth=1, linestyle=:dot, color=:red)
    hline!(p2, [0], label="", color=:black, alpha=0.4)

    method_labels = ["Naive", "Block\n(simple)", "Block from tau \n (simple)", "Block\n(FP)", "Block from tau \n (FP)", "ACF\n(FFT)", "ACF\n(loop)"]
    method_values = [stats.naive_error, block.true_error_estimate_blockAvg, stats.error_tau_block, block.optimal_error_fp, stats.error_tau_block_fp, stats.error_acf, stats.error_acf_loop]
    p3 = bar(method_labels, method_values, legend=false, ylabel="Estimated std. error",
    title="Error Estimate Comparison")
    
    p4 = plot((0:N-1) .* dt, stats.acf_abs_diff, label="Acf-loop and Acf-fft", linewidth=2, xlabel="Lag (ps)", ylabel="diff", title="ACF: FFT vs double loop")
    
    combined = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900),
                    plot_title="Method Comparison: $name ($ensemble_name)", margin=5Plots.mm)
    savefig(combined, "comparison_$(ensemble_name)_$(name).svg")
end
L_dt = 2.0 # in femtoseconds 
s_ene = 100 
dt = s_ene * L_dt * 1e-3  # in picoseconds

# NVT
col_names_nvt = ["Temperature", "Kinetic_Energy", "Potential_Energy", "Conserved_Energy", "Pressure"]
col_indices_nvt = [3, 4, 5, 6, 7]
process_ensemble("energy_NVT.txt", "NVT", col_names_nvt, col_indices_nvt, dt)

# NpT 
col_names_npt = ["Temperature", "Kinetic_Energy", "Potential_Energy", "Conserved_Energy", "Pressure", "Density"]
col_indices_npt = [3, 4, 5, 6, 7, 8]
process_ensemble("energy_NpT.txt", "NpT", col_names_npt, col_indices_npt, dt)

#= 
- chybi porovnani τ ? 
- acf potrebuje nekde uriznout na pocitani chyby - mozna vyresi vaga? (pymbar)
- nejak upravit ten bar graf tech chyb 
- udelat for loop for calc_error ..() 
- prohlednout jestli vracim a davam jenom potrebne promenne 
- init struktura z parametru simulace ? 
- jeste mozna prodlouzit simulaci ? 
- bylo by fajn jeste udelat porovnani chyb u stejne metody ale jineho dohadu chyb 
- asi se zbavit FP metody, protoze na tu asi nemam dost dlouhe data 
- chybi porovnani NVT a NpT souboru 
- chybi analyza chyby a pocitani heat capacity
=# 