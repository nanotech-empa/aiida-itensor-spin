module AuxFunctions

# packages #
using ITensorMPS
using Random
using WignerSymbols
using LinearAlgebra
###############################################################################

# function that creates a spin state from s,N,Sz
## GC, 22-04-2026
function spinstate_sNSz(s,N,Sz;random="yes")
    #=
       function that creates a spin state from s,N,Sz

       Notes:
       - as of now, only s=1/2 and s=1 are possible
       - for s=1, the state created has minimal nr of "Z0"
    =#

    # s=1/2
    if s==1/2
        if N%2 == abs(2*Sz)%2 && abs(Sz) <= s*N
            state = ["Dn" for i in 1:N]
            for i in 1:Int(N/2+Sz)
                state[i] = "Up"
            end
            if random=="yes"
                shuffle!(state)
            end
        else
            error("ERROR: check s, N and Sz")
        end
    # s=1
    elseif s==1
        if (2*Sz)%2 == 0 && abs(Sz) <= s*N
            state = ["Dn" for i in 1:N]
            if N%2==0
                for i in 1:Int(N/2+floor(Sz/2))
                    state[i] = "Up"
                end
                if abs(Sz%2)==1
                    state[Int(N/2+floor(Sz/2))+1] = "Z0"
                end
            elseif N%2==1
                for i in 1:Int(floor(N/2)+ceil(Sz/2))
                    state[i] = "Up"
                end
                if abs(Sz%2)==0
                    state[Int(floor(N/2)+ceil(Sz/2))+1] = "Z0"
                end
            end
            if random=="yes"
                shuffle!(state)
            end
        else
            error("ERROR: check s, N and Sz")
        end
    else
        error("ERROR: as of now, only s=1/2 and s=1 are possible")
    end

    return(state)
end

# function that creates a fermion state from No,Ne,Sz
## GC, 22-04-2026
function fermionstate_NoNeSz(No,Ne,Sz;random="yes")
    #=
       function that creates a fermion state from No,Ne,Sz
    =#

    Ne_up,Ne_dn = 0,0
    if Ne%2 == 0 && (2*Sz)%2 == 0
		Ne_up = Int(Ne÷2 + Sz)
		Ne_dn = Int(Ne÷2 - Sz)
	elseif Ne%2 == 1 && abs(2*Sz)%2 == 1
		Ne_up = Int(Ne÷2 + (Sz+1/2))
		Ne_dn = Int(Ne÷2 - (Sz-1/2))
    else
        error("ERROR: check Ne and Sz")
    end

    state = ["Emp" for i in 1:No]
    if Ne <= No
        for i in 1:Ne_up
            state[i] = "Up"
        end
        for i in (Ne_up+1):Ne
            state[i] = "Dn"
        end
    elseif Ne > No
        for i in 1:(Ne-No)
            state[i] = "UpDn"
        end
        for i in (Ne-No+1):Ne_up
            state[i] = "Up"
        end
        for i in Ne_up+1:No
            state[i] = "Dn"
        end
    end
    if random=="yes"
        shuffle!(state)
    end

    return(state)
end

"""
Uses Wigner-Eckart theorem to compute reduced matrix element, following 
Sakurai convention.

Given 
    Tcalc = <j,m| T(r,q) |jp,mp>,
where T(r,q) is q-component of spherical tensor of rank r, and |j,m>, |jp,mp>
are angular momentum eigenstates, reduced matrix element is 
    Tred = sqrt(2*j+1) / <jp,mp; r,q| j,m> * Tcalc,
where <jp,mp; r,q| j,m> ≠ 0 is Clebsch-Gordan coefficient.

Notes
-----
All checks on parameters, e.g. integer or half-integer character of j, are 
done within tolerance <tol> (default: 1e-5).

GC, 13-07-2026
"""
function Tred_WignerEckart(
    j::Real, m::Real, r::Integer, q::Integer, jp::Real, mp::Real, 
    Tcalc::Number; 
    tol::Real=1e-5
    )::Number

	# checks
    ## j
	if j < 0 || !isapprox(2*j, round(2*j); atol=tol, rtol=0)
		throw(ArgumentError("j must be non-negative integer or half-integer."))
    else
        j  = round(2*j)/2
    end
    ## m
    if !isapprox(2*m, round(2*m); atol=tol, rtol=0)
		throw(ArgumentError("m must be integer or half-integer."))
    else
        m = round(2*m)/2
	end
	if !(m in -j:1:j)
		throw(ArgumentError("m must be in {-j, -j+1, ..., j}."))
	end
    ## r
    if r < 0
		throw(ArgumentError("r must be a non-negative integer."))
	end
    ## q
	if !(q in -r:1:r)
		throw(ArgumentError("q must be in {-r, -r+1, ..., r}."))
	end
    # jp
	if jp < 0 || !isapprox(2*jp, round(2*jp); atol=tol, rtol=0)
		throw(ArgumentError("jp must be non-negative integer or "*
            "half-integer."))
    else
        jp = round(2*jp)/2
	end
    ## mp
    if !isapprox(2*mp, round(2*mp); atol=tol, rtol=0)
		throw(ArgumentError("mp must be integer or half-integer."))
    else
        mp = round(2*mp)/2
	end
	if !(mp in -jp:1:jp)
		throw(ArgumentError("mp must be in {-jp, -jp+1, ..., jp}."))
	end
	
	# Clebsch-Gordan coefficient
	cg = Float64(clebschgordan(jp,mp, r,q, j,m))
    ## check 
    if isapprox(cg, 0)
        throw(ArgumentError("Clebsch-Gordan coefficient must be nonzero."))
    end

	# Tred
	Tred = sqrt(2*j+1)/cg * Tcalc

	return Tred
end

"""
Symmetrize (subset of) eigenstates ψn with respect to symmetry operator O.

Notes
-----
- For proper symmetrization, subset must not lack any degenerate state.
- After obtaining O in the truncated basis spanned by subset of eigenstates, 
checks if it is Hermitian within tolerance <tol> (default: 1e-5).

GC, 13-07-2026
"""
function symmetrize_eigenstates(
    ψn::Vector{MPS}, O::MPO;
    tol::Real=1e-5
    )::Tuple{Vector,Vector{MPS}}

    # O in basis spanned by subset of eigenstates (truncated basis of H)
    """
    O_ij = <i|O|j> 
    """
    dim = length(ψn)
    O_trunc = zeros(dim,dim)
    for i in 1:dim 
        for j in 1:dim
            O_trunc[i,j] = inner(ψn[i]',O,ψn[j])
        end
    end

    # diagonalization of O in truncated basis
    if isapprox(O_trunc, O_trunc'; atol=tol, rtol=0)
        vals, vecs = eigen(Hermitian(O_trunc))
    else
        vals, vecs = eigen(O_trunc)
    end

    # symmetrized eigenstates
    ψn_sym = Vector{MPS}(undef, dim)
    for i in 1:dim
        ψn_sym[i] = sum(ψn[j] * vecs[j,i] for j in 1:dim)
    end

    return vals, ψn_sym
end

"""
Calculates Boltzmann distribution:

    Pn = wn/Z,
        wn = exp(-beta*En),
        Z = sum_n wn

Notes
-----
- To improve numerical stability, energies are shifted by min(En).
- If gn is provided, En are assumed to be unique eigenvalues with degeneracies
gn, so that wn = gn * exp(-beta*En).
- Calculation is approximated unless En includes all energies.

GC, 10-07-2026
"""
function calc_Boltzmann(
    beta::Float64, En::Vector{Float64}; 
    gn::Union{Nothing,Vector{Float64}}=nothing
    )::Vector{Float64}
    
    # checks
    if beta <= 0
        throw(ArgumentError("beta must be positive."))
    end
    if gn !== nothing && length(gn) != length(En)
        throw(ArgumentError("gn must have same length as En."))
    end

    # shift energies
    En_shifted = En .- minimum(En)

    # degeneracies, gn
    if gn === nothing
        gn = ones(length(En))
    end

    # Boltzmann weights, wn
    wn = gn .* exp.(-beta .* En_shifted)

    # partition function, Z
    Z = sum(wn)

    # Boltzmann distribution, Pn
    Pn = wn ./ Z

    return Pn
end

end