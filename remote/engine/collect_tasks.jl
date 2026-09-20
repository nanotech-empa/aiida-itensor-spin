###############################################################################
# Validate and deterministically collect generic (state, site, channel) tasks.
###############################################################################
using JLD2
using SHA
using ITensors
using ITensorMPS

const CHANNELS = ("zz", "pm", "mp")
const CHANNEL_WEIGHTS = Dict("zz" => 1.0, "pm" => 0.5, "mp" => 0.5)
const FORMAT_VERSION = 3
const MOMENT_CONVENTION = "chebyshev-2n-minus-1-v2"

function parse_options(args)
    options = Dict{String,String}()
    index = 1
    while index <= length(args)
        flag = args[index]
        startswith(flag, "--") || error("Unexpected positional argument: $flag")
        index == length(args) && error("Missing value for $flag")
        options[flag] = args[index + 1]
        index += 2
    end
    return options
end

function required_option(options, name)
    value = get(options, name, "")
    isempty(value) && error("Required option: $name VALUE")
    return value
end

function main(args)
options = parse_options(args)
input_root = abspath(required_option(options, "--input-root"))
output_dir = abspath(required_option(options, "--output"))
checkpoint_filename = abspath(required_option(options, "--checkpoint"))
backend = lowercase(required_option(options, "--backend"))
backend in ("full", "streaming") ||
    error("Unsupported --backend $backend; choose full or streaming")
threading_mode = lowercase(required_option(options, "--threading-mode"))
threading_mode in ("outer", "blocksparse") ||
    error("Unsupported --threading-mode $threading_mode; choose outer or blocksparse")
reference_filename = get(options, "--reference", "")
reference_tolerance =
    parse(Float64, get(options, "--reference-tolerance", "1e-10"))
spectral_cutoff_override =
    haskey(options, "--spectral-cutoff") ?
    parse(Float64, options["--spectral-cutoff"]) : nothing
spectral_cutoff_override === nothing || spectral_cutoff_override > 0 ||
    error("--spectral-cutoff must be positive")
mkpath(output_dir)

checkpoint_hash = bytes2hex(open(sha256, checkpoint_filename))
checkpoint = load(checkpoint_filename)
checkpoint_cutoff = get(checkpoint, "spectral_cutoff", checkpoint["cutoff"])
spectral_cutoff = spectral_cutoff_override === nothing ?
    checkpoint_cutoff : spectral_cutoff_override
L = checkpoint["L"]
length(checkpoint["sites"]) == L ||
    error("Checkpoint inconsistency: L=$L but length(sites) differs")
nstates = length(checkpoint["ψn_sym"])
expected_task_count = 3 * L * nstates

task_files = String[]
for (root, _, files) in walkdir(input_root)
    for filename in files
        startswith(filename, "task_") && endswith(filename, ".jld2") &&
            push!(task_files, joinpath(root, filename))
    end
end
sort!(task_files)
length(task_files) == expected_task_count ||
    error(
        "Found $(length(task_files)) task files; expected $expected_task_count " *
        "for states=$nstates, L=$L, channels=$(length(CHANNELS))",
    )

spectra = Dict{Tuple{Int,Int,String},Vector{Float64}}()
task_rows = NamedTuple[]
common_omega = nothing
for filename in task_files
    data = load(filename)
    data["format_version"] == FORMAT_VERSION ||
        error("Unsupported format in $filename")
    get(data, "moment_convention", "") == MOMENT_CONVENTION ||
        error("Unexpected moment convention in $filename")
    data["backend"] == backend ||
        error("Unexpected backend in $filename: $(data["backend"])")
    data["threading_mode"] == threading_mode ||
        error("Unexpected threading mode in $filename: $(data["threading_mode"])")
    data["checkpoint_hash"] == checkpoint_hash ||
        error("Checkpoint hash mismatch in $filename")
    task_cutoff = get(data, "spectral_cutoff", checkpoint_cutoff)
    task_cutoff == spectral_cutoff ||
        error("Spectral-cutoff mismatch in $filename: $task_cutoff")
    data["L"] == L || error("L mismatch in $filename")
    data["nstates"] == nstates || error("State-count mismatch in $filename")

    state_index = data["state_index"]
    site_index = data["site_index"]
    channel = data["channel"]
    1 <= state_index <= nstates || error("Invalid state in $filename")
    1 <= site_index <= L || error("Invalid site in $filename")
    channel in CHANNELS || error("Invalid channel in $filename")
    data["channel_weight"] == CHANNEL_WEIGHTS[channel] ||
        error("Channel-weight mismatch in $filename")

    key = (state_index, site_index, channel)
    haskey(spectra, key) && error("Duplicate task $key in $filename")
    spectrum = Vector{Float64}(data["spectrum"])
    omega = Vector{Float64}(data["omega"])
    length(spectrum) == length(omega) ||
        error("Spectrum/grid length mismatch in $filename")
    all(isfinite, spectrum) || error("Non-finite spectrum in $filename")
    all(isfinite, omega) || error("Non-finite frequency grid in $filename")
    if common_omega === nothing
        common_omega = omega
    else
        omega == common_omega ||
            error("Frequency grid differs in $filename")
    end
    spectra[key] = spectrum
    push!(
        task_rows,
        (
            task_id=data["task_id"],
            state=state_index,
            site=site_index,
            channel=channel,
            weight=data["channel_weight"],
            worker=data["worker_rank"],
            threads=data["worker_threads"],
            threading_mode=data["threading_mode"],
            seconds=data["task_seconds"],
            file=filename,
        ),
    )
end

expected_keys = Set(
    (state_index, site_index, channel)
    for state_index in 1:nstates
    for site_index in 1:L
    for channel in CHANNELS
)
found_keys = Set(keys(spectra))
found_keys == expected_keys || begin
    missing = sort!(collect(setdiff(expected_keys, found_keys)))
    unexpected = sort!(collect(setdiff(found_keys, expected_keys)))
    error("Task coverage mismatch: missing=$missing unexpected=$unexpected")
end

# Preserve the original summation order exactly: zz + 0.5*pm + 0.5*mp.
χGSi = [Vector{Vector{Float64}}(undef, L) for _ in 1:nstates]
for state_index in 1:nstates
    for site_index in 1:L
        χzz = spectra[(state_index, site_index, "zz")]
        χpm = spectra[(state_index, site_index, "pm")]
        χmp = spectra[(state_index, site_index, "mp")]
        length(χzz) == length(χpm) == length(χmp) ||
            error("Channel-grid size mismatch for state=$state_index site=$site_index")
        χGSi[state_index][site_index] = χzz .+ 0.5 .* χpm .+ 0.5 .* χmp
    end
end

En = checkpoint["En"]
Sn = checkpoint["Sn"]
Szn = checkpoint["Szn"]
Emax = checkpoint["Emax"]
En_sym = checkpoint["En_sym"]
Sn_sym = checkpoint["Sn_sym"]
Szn_sym = checkpoint["Szn_sym"]
seed = checkpoint["seed"]
merged_filename = joinpath(output_dir, "chi_L$(L).jld2")
jldsave(
    merged_filename;
    χGSi,
    L,
    En,
    Sn,
    Szn,
    Emax,
    En_sym,
    Sn_sym,
    Szn_sym,
    seed,
    checkpoint_filename,
    checkpoint_hash,
    spectral_cutoff,
    task_rows,
    backend,
    threading_mode,
    moment_convention=MOMENT_CONVENTION,
)

omega = common_omega::Vector{Float64}
open(joinpath(output_dir, "spectrum.tsv"), "w") do io
    write(io, "state\tsite\tomega_meV\tchi\n")
    for state_index in 1:nstates
        for site_index in 1:L
            for frequency_index in eachindex(omega)
                write(
                    io,
                    "$state_index\t$site_index\t$(omega[frequency_index])\t" *
                    "$(χGSi[state_index][site_index][frequency_index])\n",
                )
            end
        end
    end
end

sort!(task_rows; by=row -> row.task_id)
open(joinpath(output_dir, "task_timings.tsv"), "w") do io
    write(io, "task\tstate\tsite\tchannel\tweight\tworker\tthreads\tthreading_mode\tseconds\tfile\n")
    for row in task_rows
        write(
            io,
            "$(row.task_id)\t$(row.state)\t$(row.site)\t$(row.channel)\t" *
            "$(row.weight)\t$(row.worker)\t$(row.threads)\t$(row.threading_mode)\t" *
            "$(row.seconds)\t" *
            "$(row.file)\n",
        )
    end
end

reference_max_abs_difference = nothing
if !isempty(reference_filename)
    reference_filename = abspath(reference_filename)
    reference = load(reference_filename, "χGSi")
    length(reference) == nstates ||
        error("Reference state count differs")
    differences = Float64[]
    for state_index in 1:nstates, site_index in 1:L
        length(reference[state_index][site_index]) ==
            length(χGSi[state_index][site_index]) ||
            error(
                "Reference grid size differs for state=$state_index " *
                "site=$site_index",
            )
        append!(
            differences,
            abs.(
                reference[state_index][site_index] .-
                χGSi[state_index][site_index],
            ),
        )
    end
    reference_max_abs_difference = maximum(differences)
end

open(joinpath(output_dir, "collection_report.txt"), "w") do io
    write(io, "format version = $FORMAT_VERSION\n")
    write(io, "backend = $backend\n")
    write(io, "threading mode = $threading_mode\n")
    write(io, "moment convention = $MOMENT_CONVENTION\n")
    write(io, "L = $L\n")
    write(io, "states = $nstates\n")
    write(io, "channels = $(collect(CHANNELS))\n")
    write(io, "expected tasks = $expected_task_count\n")
    write(io, "collected tasks = $(length(task_files))\n")
    write(io, "checkpoint = $checkpoint_filename\n")
    write(io, "checkpoint sha256 = $checkpoint_hash\n")
    write(io, "spectral cutoff = $spectral_cutoff\n")
    if reference_max_abs_difference !== nothing
        write(io, "reference = $reference_filename\n")
        write(
            io,
            "reference max abs difference = $reference_max_abs_difference\n",
        )
        write(io, "reference tolerance = $reference_tolerance\n")
    end
end

println("Merged result written to $merged_filename")
println(
    "Validated $expected_task_count tasks: $nstates states × $L sites × " *
    "$(length(CHANNELS)) channels",
)
if reference_max_abs_difference !== nothing
    println("Reference max abs difference = $reference_max_abs_difference")
    reference_max_abs_difference <= reference_tolerance ||
        error(
            "Reference difference $reference_max_abs_difference exceeds " *
            "tolerance $reference_tolerance",
        )
end
end

main(ARGS)
