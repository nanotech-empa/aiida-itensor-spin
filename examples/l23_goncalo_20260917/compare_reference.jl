using JLD2, ITensors, ITensorMPS, LinearAlgebra, TOML

root = ARGS[1]
work = length(ARGS) == 2 ? ARGS[2] : root
reference = load(joinpath(root, "reference", "chi_L23.jld2"), "χ_n_i")
result = load(joinpath(work, "state", "merged", "chi_L23.jld2"), "χGSi")
gs_path = length(ARGS) == 2 ? joinpath(work, "gs_checkpoint.jld2") : joinpath(work, "gs", "gs_checkpoint.jld2")
gs = load(gs_path)
@assert length(reference) == length(result) == 1
@assert length(reference[1]) == length(result[1]) == 23
@assert gs["JL"] == 16.7 && gs["JR"] == 51.1
@assert gs["cutoff"] == 1e-8 && gs["spectral_cutoff"] == 1e-6
@assert gs["ω_smearing"] == 1.5

function integral(y, x)
    return sum((y[1:end-1] .+ y[2:end]) .* diff(x)) / 2
end

all_ref, all_result = Float64[], Float64[]
max_area_relative_difference = 0.0
omega = collect(range(0.0, 100.0; length=333))
open(joinpath(work, "comparison_by_site.tsv"), "w") do io
    println(io, "site\tmax_abs_difference\trelative_L2\treference_integral\tresult_integral\treference_peak_meV\tresult_peak_meV")
    for site in 1:23
        a, b = reference[1][site], result[1][site]
        @assert length(a) == length(b) == length(omega)
        @assert all(isfinite, a) && all(isfinite, b)
        append!(all_ref, a)
        append!(all_result, b)
        area_a, area_b = integral(a, omega), integral(b, omega)
        relative_area = abs(area_b - area_a) / max(abs(area_a), eps())
        global max_area_relative_difference = max(max_area_relative_difference, relative_area)
        println(io, join((site, maximum(abs.(b .- a)), norm(b .- a)/max(norm(a), eps()),
                         area_a, area_b, omega[argmax(a)], omega[argmax(b)]), '\t'))
    end
end

report = Dict(
    "reference_key" => "χ_n_i", "result_key" => "χGSi",
    "spectral_tasks" => 69, "sites" => 23, "frequency_points" => 333,
    "max_abs_difference" => maximum(abs.(all_result .- all_ref)),
    "relative_L2_difference" => norm(all_result .- all_ref) / norm(all_ref),
    "max_site_relative_integral_difference" => max_area_relative_difference,
    "reference_ground_energy_meV" => -574.5775840718105,
    "ground_energy_meV" => minimum(gs["En_sym"]),
    "ground_energy_abs_difference_meV" => abs(minimum(gs["En_sym"]) + 574.5775840718105),
    "reference_highest_energy_meV" => 401.89997937127185,
    "highest_energy_meV" => gs["Emax"],
    "note" => "Independent random initial MPS and runtime; report numerical differences, not bitwise equality. No acceptance tolerance has been assumed.",
)
open(joinpath(work, "comparison.toml"), "w") do io
    TOML.print(io, report; sorted=true)
end
TOML.print(stdout, report; sorted=true)
