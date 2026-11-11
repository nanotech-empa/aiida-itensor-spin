###############################################################################
# packages #
###############################################################################
using Dates
using LinearAlgebra
using ITensorMPS
using JLD2
using Random

###############################################################################
# my libraries #
###############################################################################
lib_dir = "/Users/scgo/Documents/Mac_Empa/my_libs/julia/ITensor/"
include(lib_dir*"/Imports.jl")

###############################################################################
# main #
###############################################################################

# physical parameters
L = 23
i_imp = ceil(Int, L/2) # site spin-1/2 impurity (middle of chain)
J = 18.4 # meV
JL = 16.7 # meV
JR = 51.1 # meV

# other parameters 
linkdim = 100 # variable to randomize initial MPS
cutoff = 1e-8 # DMRG cutoff
precE = 1e-5 # precision in energy, meV
precEmax = 1e-3 # precision in energy for highest-energy state, meV
nexc = 2 # nr of excited states
w = 1000 # penalty for non-orthogonality in excited states, meV
T = 4 # temperature, K 
tol_sym = 1e-2 # tolerance to check symmetrization
ωmax = 100.0 # maximum frequency, meV
try_doubleN = true # flag to try doubling nr of Chebyshev polynomials
ω_smearing = 1.5 # frequency smearing, meV
cutoff_χ = 1e-6 # cutoff for Chebyshev expansion

# start time
start_time = DateTime(now())

# set 1 thread in BLAS operations for better performance
BLAS.set_num_threads(1)

# open file
fw = open("L$(L)__impurity_site$(i_imp).txt", "w")

# sites 
sites = siteinds("S=1",L;conserve_sz=true)
sites[i_imp] = siteind("S=1/2";conserve_sz=true)

# Hamiltonian 
H = Hamiltonians.H_Heis_impurity_1D_OBC(J,JL,JR,i_imp,sites)

# S^2 operator 
S2 = Operators.S2_op(sites)

# Sz operator 
Sz = Operators.Sz_op(sites)

# Sz(i), S+(i), S-(i) operators
Szi = [Operators.Szi_op(i,sites) for i in 1:L]
Spi = [Operators.Spi_op(i,sites) for i in 1:L]
Smi = [Operators.Smi_op(i,sites) for i in 1:L]

# initial state
"""
Sz = 1/2
"""
statei = AuxFunctions.spinstate_sNSz(1,L-1,0)
insert!(statei, i_imp, "Up")
ψi = randomMPS(sites,statei,linkdim)

# run DMRG
## groundstate
E0,ψ0 = DMRGsweeps.DMRGmaxdim_convE(H,ψi,precE,cutoff=cutoff)
println()
## excited states
En = [E0]
ψn = [ψ0]
for n in 1:nexc
    Eaux,ψaux = DMRGsweeps.DMRGmaxdim_convE(H,ψi,precE,ψn=ψn,w=w,
        cutoff=cutoff)
    println()
    push!(En,Eaux)
    push!(ψn,ψaux)
end
## sort (in case DMRG fails in order)
idx = sortperm(En)
En = En[idx]
ψn = ψn[idx]
## write energies 
write(fw, "# List of energies:\n")
write(fw, string(En), "\n")
write(fw, "\n")
## highest-energy state
E0_mH,_ = DMRGsweeps.DMRGmaxdim_convE(-H,ψi,precEmax,cutoff=cutoff)
Emax = -E0_mH
println()
## write energy
write(fw, "# Energy of highest-energy state:\n")
write(fw, string(Emax), "\n")
write(fw, "\n")

# <psin|S^2|psin>
S2n = [inner(ψn[n]',S2,ψn[n]) for n in 1:length(En)]
## corresponding S
Sn = (-1 .+ sqrt.(1 .+ 4*S2n)) / 2
## write S
write(fw, "# List of S:\n")
write(fw, string(Sn), "\n")
write(fw, "\n")

# <psin|Sz|psin>
Szn = [inner(ψn[n]',Sz,ψn[n]) for n in 1:length(En)]
## write
write(fw, "# List of Sz:\n")
write(fw, string(Szn), "\n")
write(fw, "\n")

# temperature in meV 
kB = 8.617333262e-2 #meV/K
T_meV = kB*T

# GS manifold
"""
If excitation energy < T/5, we consider this state as GS
"""
nGS = count(E -> E - En[1] <= T_meV/5, En)

# write start of symmetrization
write(fw, "========== Symmetrization (GS manifold) ==========\n")
write(fw, "\n")

# symmetrize GS manifold wrt to S^2
ψn_GS = ψn[1:nGS]
S2n_sym, ψn_sym = AuxFunctions.symmetrize_eigenstates(ψn_GS,S2)

# energies after symmetrization 
En_sym = [inner(ψn_sym[n]',H,ψn_sym[n]) for n in 1:length(ψn_sym)]
## sort by energy
idx = sortperm(En_sym)
En_sym = En_sym[idx]
ψn_sym = ψn_sym[idx]
S2n_sym = S2n_sym[idx]
## write 
write(fw, "# List of energies after symmetrization:\n")
write(fw, string(En_sym), "\n")
write(fw, "\n")

# S after symmetrization
Sn_sym = (-1 .+ sqrt.(1 .+ 4*S2n_sym)) / 2
## write
write(fw, "# List of S after symmetrization:\n")
write(fw, string(Sn_sym), "\n")
write(fw, "\n")

# Sz after symmetrization 
Szn_sym = [inner(ψn_sym[n]',Sz,ψn_sym[n]) for n in 1:length(ψn_sym)]
## write
write(fw, "# List of Sz after symmetrization:\n")
write(fw, string(Szn_sym), "\n")
write(fw, "\n")

# check S and Sz are integer of half-integer (within tol_sym)
Sn_sym_round = round.(2 .* Sn_sym) ./ 2
@assert all(isapprox.(Sn_sym, Sn_sym_round; atol=tol_sym, rtol=0))
Szn_sym_round = round.(2 .* Szn_sym) ./ 2
@assert all(isapprox.(Szn_sym, Szn_sym_round; atol=tol_sym, rtol=0))

# write start of χ calculation
write(fw, "========== χ[nGS][i] ==========\n")
write(fw, "\n")
## print relevant info
println(stdout, "==================================================\n")
println(stdout, "Calculating χ, nr of GS = $nGS")
bandwidth = @isdefined(Wstar) ? Wstar : Emax - minimum(En_sym)
println(stdout, "Effective nr of Chebyshev polynomials "*
    "~ $(round(Int, bandwidth/ω_smearing))\n")
flush(stdout)

# χ for (symmetrized) states in GS manifold
## GS energy
E0_sym = minimum(En_sym)
## tasks to thread
tasks = [(n, i, type) for n in 1:nGS for i in 1:L for type in (:zz, :pm, :mp)]
shuffle!(tasks)
## initialize χzz, χ+-, χ-+ 
χzz_n_i = [Vector{Vector{Float64}}(undef, L) for n in 1:nGS]
χpm_n_i = [Vector{Vector{Float64}}(undef, L) for n in 1:nGS]
χmp_n_i = [Vector{Vector{Float64}}(undef, L) for n in 1:nGS]
## threaded loop
next_task = Threads.Atomic{Int}(1)
Threads.@threads for _ in 1:Threads.nthreads() #threads defined in run script
    while true
        k = Threads.atomic_add!(next_task, 1)
        #this assigns k=next_task and then adds 1 to next_task
        
        if k > length(tasks)
            break
        end

        n, i, type = tasks[k]
        ψ0n = ψn_sym[n]
        print_prefix = "n=$n, i=$i, χ$type: "

        println(stdout, "n=$n, i=$i, χ$type started")
        flush(stdout)

        ### χ_zz
        if type == :zz
            B = Szi[i]
            C = Szi[i]
            χzz_n_i[n][i] = SpectralFunction.spectral_function(
                ψ0n, H, B, C, E0_sym, ωmax, sites,
                cutoff=cutoff_χ, 
                Emax=Emax,
                try_doubleN=try_doubleN,
                ω_smearing=ω_smearing,
                print_prefix=print_prefix
            )

        ### χ_+-
        elseif type == :pm
            B = Spi[i]
            C = Smi[i]
            χpm_n_i[n][i] = SpectralFunction.spectral_function(
                ψ0n, H, B, C, E0_sym, ωmax, sites,
                cutoff=cutoff_χ, 
                Emax=Emax,
                try_doubleN=try_doubleN,
                ω_smearing=ω_smearing,
                print_prefix=print_prefix
            )

        ### χ_-+
        elseif type == :mp
            B = Smi[i]
            C = Spi[i]
            χmp_n_i[n][i] = SpectralFunction.spectral_function(
                ψ0n, H, B, C, E0_sym, ωmax, sites,
                cutoff=cutoff_χ, 
                Emax=Emax,
                try_doubleN=try_doubleN,
                ω_smearing=ω_smearing,
                print_prefix=print_prefix
            )
        end

        println(stdout, "n=$n, i=$i, χ$type finished")
        flush(stdout)
    end
end
## χ
χ_n_i = χzz_n_i .+ 0.5 .* χpm_n_i .+ 0.5 .* χmp_n_i
## write
filename = "chi_L$(L).jld2"
@save filename χ_n_i
write(fw, "χ saved as $filename\n")
write(fw, "\n")

# write total time 
write(fw, "----------\n")
write(fw,"total time = "*
    string( Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(now())
    - Dates.DateTime(start_time))) ) * "\n")

# close file
close(fw)