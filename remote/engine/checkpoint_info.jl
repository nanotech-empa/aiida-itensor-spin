###############################################################################
# Print the dimensions of a GS checkpoint without any L-specific assumptions.
###############################################################################
using ITensors
using ITensorMPS
using JLD2

length(ARGS) == 1 || error("Usage: checkpoint_info.jl GS_CHECKPOINT")
checkpoint_filename = abspath(ARGS[1])
isfile(checkpoint_filename) || error("Checkpoint not found: $checkpoint_filename")

checkpoint = load(checkpoint_filename)
for key in ("L", "sites", "ψn_sym")
    haskey(checkpoint, key) || error("Checkpoint is missing required key: $key")
end

L = checkpoint["L"]
length(checkpoint["sites"]) == L ||
    error("Checkpoint inconsistency: L=$L but length(sites)=$(length(checkpoint["sites"]))")
nstates = length(checkpoint["ψn_sym"])
nstates >= 1 || error("Checkpoint contains no symmetrized ground states")
tasks = 3 * L * nstates

println("$L\t$nstates\t$tasks")
