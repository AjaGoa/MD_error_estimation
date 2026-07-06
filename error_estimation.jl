using Statistics
using StatsBase
using Plots
using DelimitedFiles
using Printf

# Understanding Molecular Simulation : From Algorithms to Applications, 5.3 Statistical Error 

# TODO :
# https://github.com/sunsistemo/chronogrator/tree/master/Block 
# https://github.com/mikhail-a-ivanov/pyLJ/tree/main/MD-data
# https://github.com/rsdefever/block_average


const filename_nvt = "argon_data_NVT.txt"
const filename_npt = "argon_data_NPT.txt"

function block_average(data)
    N = length(data)
    block_sizes = Int[]
    block_errors = Float64[]
    
    # Test block sizes up to N/10
    for B in 1:floor(Int, N/10)
        n_blocks = N ÷ B # integer division (ignores the reminder) 
        if n_blocks < 5
            break 
        end
        
        blocks = [mean(data[(i-1)*B+1 : i*B]) for i in 1:n_blocks] 
        
        push!(block_sizes, B)
        # Standard error of the block means
        push!(block_errors, std(blocks) / sqrt(n_blocks))
    end
    return block_sizes, block_errors
end

function compute_stats(data)
    N = length(data)
    μ = mean(data)
    σ = std(data)
    
    # avoid division by zero in ACF
    if σ < 1e-10
        return μ, σ, 0.0, Int[], Float64[], 0.0, Float64[], 0, 0.0, 1.0, 0.0 # skips the ACF and block averaging calculations and just returns zeros and empty arrays for the rest of the values 
    end
    
    naive_error = σ / sqrt(N)
    
    # Block Averaging 
    sizes, errors_block = block_average(data)
    est_err_block = isempty(errors_block) ? 0.0 : maximum(errors_block) # "quick-and-dirty"  - to find the plateau of the block average error curve - careful, as can be oscillating and therefore overestimate the true error
    
    # Autocorrelation 
    max_lag = floor(Int, N/5)
    acf = autocor(data, 0:max_lag)
    
    # Find where ACF roughly hits zero to avoid integrating long-tail noise
    cutoff_idx = findfirst(x -> x <= 0.05, acf)
    if isnothing(cutoff_idx) # if no value is found, use the full length of acf
        cutoff_idx = length(acf)
    end
    
    # Integrated autocorrelation time (τ) 
    τ = sum(acf[2:cutoff_idx]) 
    
    # Statistical inefficiency (s = 1 + 2τ)
    s = 1 + 2 * τ
    
    # Correlated Error Estimate
    error_acf = σ * sqrt(s / N)
    
    return μ, σ, naive_error, sizes, errors_block, est_err_block, acf, cutoff_idx, τ, s, error_acf
end

function generate_subplots(sizes, errors_block, acf, cutoff_idx, naive_error, var_name, ensemble)

    p1 = plot(sizes, errors_block, 
              title="$ensemble\nBlock Averaging", 
              xlabel="Block Size (B)", 
              ylabel="Standard Error", 
              label="Block Error",
              linewidth=2, legend=:bottomright)
    hline!(p1, [naive_error], label="Naive Error", linestyle=:dash, color=:red)

    p2 = plot(0:cutoff_idx-1, acf[1:cutoff_idx], 
              title="$ensemble\nAutocorrelation", 
              xlabel="Lag time (steps)", 
              ylabel="C(t)", 
              label="ACF",
              linewidth=2)
    hline!(p2, [0], label="", color=:black, alpha=0.5)

    return p1, p2
end

function format_val(val, is_constant)
    return is_constant ? "Constant" : @sprintf("%.6f", val)
end

data_nvt, header_nvt = readdlm(filename_nvt, header=true)
data_npt, header_npt = readdlm(filename_npt, header=true)

var_names = header_nvt[1, 2:end] 
N_steps = size(data_nvt, 1)

for (i, var_name) in enumerate(var_names)
    println("\n=====================================================================")
    println(" VARIABLE: $var_name")
    println("=====================================================================")
    
    # Extract data columns (i+1 because column 1 is 'Step')
    vec_nvt = convert(Vector{Float64}, data_nvt[:, i+1])
    vec_npt = convert(Vector{Float64}, data_npt[:, i+1])
    
    # Compute statistics
    stats_nvt = compute_stats(vec_nvt)
    stats_npt = compute_stats(vec_npt)
    
    is_const_nvt = stats_nvt[2] < 1e-10 # std deviation check
    is_const_npt = stats_npt[2] < 1e-10

    # --- Terminal Output Comparison ---
    @printf("%-25s | %-20s | %-20s\n", "Metric", "NVT", "NPT") # % - signals starf of format specifier, - left-align, 25 - column of 25 characters, s - string, .6 - 6 decimal places, f - floating point
    @printf("--------------------------|----------------------|----------------------\n")
    @printf("%-25s | %-20.6f | %-20.6f\n", "Mean", stats_nvt[1], stats_npt[1])
    @printf("%-25s | %-20.6f | %-20.6f\n", "Std Dev", stats_nvt[2], stats_npt[2])
    @printf("%-25s | %-20s | %-20s\n", "Naive Error", format_val(stats_nvt[3], is_const_nvt), format_val(stats_npt[3], is_const_npt))
    @printf("--------------------------|----------------------|----------------------\n")
    @printf("%-25s | %-20s | %-20s\n", "Est. Error (Block Avg)", format_val(stats_nvt[6], is_const_nvt), format_val(stats_npt[6], is_const_npt))
    @printf("%-25s | %-20s | %-20s\n", "Est. Error (ACF)", format_val(stats_nvt[11], is_const_nvt), format_val(stats_npt[11], is_const_npt))
    
    # Compare Block Avg vs ACF Methods
    diff_nvt = is_const_nvt ? "N/A" : @sprintf("%+.6f", stats_nvt[6] - stats_nvt[11]) # @printf (Print Format): Formats the text and prints it directly to terminal - does not return any data, @sprintf (String Print Format): Formats the text and returns it as a String variable, without printing it to the screen
    diff_npt = is_const_npt ? "N/A" : @sprintf("%+.6f", stats_npt[6] - stats_npt[11])
    @printf("%-25s | %-20s | %-20s\n", "Diff (Block - ACF)", diff_nvt, diff_npt)
    
    @printf("--------------------------|----------------------|----------------------\n")
    @printf("%-25s | %-20s | %-20s\n", "Autocorr. Time (τ)", format_val(stats_nvt[9], is_const_nvt), format_val(stats_npt[9], is_const_npt))
    @printf("%-25s | %-20s | %-20s\n", "Stat Inefficiency (s)", format_val(stats_nvt[10], is_const_nvt), format_val(stats_npt[10], is_const_npt))
    println()

    p_nvt_block, p_nvt_acf = generate_subplots(stats_nvt[4:5]..., stats_nvt[7:8]..., stats_nvt[3], var_name, "NVT") # ... - unpacks the tuple returned by compute_stats into the arguments of generate_subplots, splat operator 
    p_npt_block, p_npt_acf = generate_subplots(stats_npt[4:5]..., stats_npt[7:8]..., stats_npt[3], var_name, "NPT")
    
    # Layout: 2 columns (NVT left, NPT right), 2 rows (Block Avg top, ACF bottom)
    combined_plot = plot(
        p_nvt_block, p_npt_block, 
        p_nvt_acf,   p_npt_acf, 
        layout = (2, 2), 
        size = (1000, 700), 
        margin = 5Plots.mm,
        plot_title = "Analysis: $var_name (NVT vs NPT)"
    )
    
    savefig(combined_plot, "error_analysis_comparison_$(var_name).png")
end

# Gemini was so kind as to provide the code for plotting and a lot of advice