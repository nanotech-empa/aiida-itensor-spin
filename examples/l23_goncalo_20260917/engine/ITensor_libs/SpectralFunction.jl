module SpectralFunction
"""
    Implementation based on https://doi.org/10.1103/PhysRevB.83.195115
    Code based on https://github.com/grafrap/DMRG_Bachelor-Arbeit/
"""

# packages and libraries #
using ITensorMPS, ITensors
using ..Operators
using Plots
using LaTeXStrings
using LinearAlgebra
###############################################################################

# Jackson damping
## GC, 22-04-2026
function Jackson_damping(N::Int)
    #=
        Jackson damping factors g_n, Eq.(15)
    =#

    gn = zeros(N)
    for n in 1:N
        gn[n] = ((N-n+1)*cos(π/(N+1)*n) + sin(π/(N+1)*n)/tan(π/(N+1))) / (N+1)
    end

    return gn
end

# Chebyshev vectors
## GC, 22-07-2026
function Chebyshev_vectors(C::MPO, H::MPO, ψ0::MPS, N::Int; 
    cutoff::Float64=1e-8, dK::Union{Int,Nothing}=nothing, 
    nsweeps::Union{Int,Nothing}=nothing, ϵP::Union{Float64,Nothing}=nothing)
    #=
        Chebyshev vectors ∣t_n⟩, Eqs.(29,30)

        Notes:
        - if dK, nsweeps, ϵP are provided, triggers energy truncation (see 
            Section III-B)
    =#

    # initialize |t_n>
    tn = Vector{MPS}(undef, N)

    # trigger energy truncation
    Etrunc_flag = false
    if dK!=nothing && nsweeps!=nothing && ϵP!=nothing
        Etrunc_flag = true
    end
    
    # |t_0>
    tn[1] = apply(C, ψ0; cutoff=cutoff)
    ## energy truncation
    if Etrunc_flag
        println("energy truncation of |t_1>")
        tn[1] = proj_highE(tn[1], H, dK, nsweeps, ϵP; cutoff=cutoff)
    end

    # |t_1>
    tn[2] = apply(H, tn[1]; cutoff=cutoff)
    ## energy truncation
    if Etrunc_flag
        println("\nenergy truncation of |t_2>")
        tn[2] = proj_highE(tn[2], H, dK, nsweeps, ϵP; cutoff=cutoff)
    end

    # |t_n>
    for n in 3:N
        tmp = 2*apply(H, tn[n-1]; cutoff=cutoff)
        tn[n] = add(tmp, -tn[n-2]; cutoff=cutoff)
        ## energy truncation
        if Etrunc_flag
            println("\nenergy truncation of |t_", n, ">")
            tn[n] = proj_highE(tn[n], H, dK, nsweeps, ϵP; cutoff=cutoff)
            if n==N 
                println("\n")
            end
        end
    end
  
    return tn
end

# Chebyshev moments
## GC, 23-04-2026
function Chebyshev_moments(ψ0::MPS, B::MPO, tn::Array{MPS,1}; 
    C::Union{MPO,Nothing}=nothing)
    #=
        Chebyshev moments μ_n, Eq.(25)

        Notes:
        - if B = C^†, N Chebyshev vectors determine 2N-1 moments,
            mu_0, ..., mu_(2N-2), Eqs.(33-34)
    =#

    N = length(tn)

    # normal routine, Eq.(25)
    if C==nothing || isapprox(B, swapprime(dag(C), 0 => 1))!=true
        μn = zeros(N)
        for n in 1:N
            μn[n] = inner(ψ0', B, tn[n])
        end

    # routine that doubles expansion, Eqs.(33-34)
    elseif isapprox(B, swapprime(dag(C), 0 => 1))==true
        Neff = 2*N - 1

        μn = zeros(Neff)

        ## μ_0
        #μn[1] = inner(ψ0', B, tn[1])
        μn[1] = inner(tn[1],tn[1])

        ## μ_1
        #μn[2] = inner(ψ0', B, tn[2])
        μn[2] = inner(tn[2],tn[1])

        ## Eq.(34) -> add 1 to all subscripts and replace n by n-1
        for n in 2:N
            μn[2*n-2] = 2*inner(tn[n],tn[n-1]) - μn[2] #repeats μn[2]
            μn[2*n-1] = 2*inner(tn[n],tn[n]) - μn[1]
        end
    end

    return μn
end

# Chebyshev polynomials
## GC, 23-04-2026
function Chebyshev_polynomials(x::Vector, N::Int)
    #=
        Chebyshev polynomials Tn, Eq.(8)
        Vectorized form
    =# 

    Tn = zeros(length(x), N)

    # T_0
    Tn[:, 1] = ones(length(x))

    # T_1
    Tn[:, 2] = x

    # T_n
    for n in 3:N
        Tn[:, n] = 2*x .* Tn[:, n-1] - Tn[:, n-2]
    end
  
    return Tn
end

# Spectral function
## GC, 21-07-2026
function spectral_function(ψ0::MPS, H::MPO, B::MPO, C::MPO, E0::Float64, 
    ωmax::Float64, sites; 
    cutoff::Float64=1e-8, ϵt::Float64=0.025, 
    Nω::Union{Int,Nothing}=nothing,
    Emax::Union{Float64,Nothing}=nothing, Wstar::Union{Float64,Nothing}=nothing,
    try_doubleN::Bool=false,
    ω_smearing::Union{Float64,Nothing}=nothing, N::Union{Int,Nothing}=nothing,
    dK::Int=30, nsweeps::Int=10, ϵP::Float64=1.0)
    #=
        Spectral function A^{BC}(ω) (herein, χ), Eq.(5)
        More details in Eqs.(20-23)

        Notes:
        - χ is computed for ω = [0,ωmax] using Nω steps (default Nω: 
            5*ωmax/ω_smearing if ω_smearing is provided or 5*ωmax*N/Wstar if N 
            is provided)
        - cutoff default (1e-8) corresponds to high precision in sum of MPOs 
            (same default as in DMRGsweeps.jl)
        - try_doubleN: if true, checks if B = C^† and if true doubles number 
            of Chebyshev polynomials at no numerical cost. Default is false 
            since it can lead to numerical imprecisions (see discussion in 
            page 6, top right)
        - if ω_smearing is provided, effective number of Chebyshev polynomials
            is ~Wstar/ω_smearing; instead, if N is provided, effective number 
            of Chebyshev polynomials is ~N
        - if Wstar is provided, triggers energy truncation (see Section III-B),
            which uses variables dK, nsweeps, ϵP 
    =#

    # checks
    if ωmax <= 0
        error("ERROR: ωmax must be > 0")
    elseif cutoff <= 0
        error("ERROR: cutoff must be > 0")
    elseif ϵt <= 0
        error("ERROR: ϵt must be > 0")
    elseif dK <= 0
        error("ERROR: dK must be > 0")
    elseif nsweeps <= 0
        error("ERROR: nsweeps must be > 0")
    elseif ϵP <= 0
        error("ERROR: ϵP must be > 0")
    end

    # effective bandwidth, Wstar
    println("# Bandwidth info")
    if Emax!=nothing && Wstar==nothing
        println("Emax provided -> calculation with full bandwidth")
        W = Emax - E0
        Wstar = W
    Etrunc_flag = false
    elseif Wstar!=nothing && Emax==nothing
        println("Wstar provided -> calculation with effective bandwidth "*
            "(energy truncation triggered)")
        Etrunc_flag = true
    elseif Emax==nothing && Wstar==nothing
        error("ERROR: provide either Emax or Wstar")
    else 
        error("ERROR: provide only Emax or Wstar, not both")
    end
    println()

    # continue checks
    if Wstar <= 0
        error("ERROR: Wstar must be > 0, check Emax or Wstar")
    elseif ωmax > Wstar
        error("ERROR: ωmax must be <= Wstar")
    end

    # number of Chebyshev polynomials
    println("# Info about Chebyshev polynomials")
    ## check if routine that doubles N will be triggered
    doubleN_flag = false
    if try_doubleN==true && isapprox(B, swapprime(dag(C), 0 => 1))==true
        doubleN_flag = true
        println("Routine that doubles number of Chebyshev polynomials "*
            "triggered (B = C^†)")
    end 
    ## effective number of Chebyshev polynomials
    if ω_smearing!=nothing && N==nothing
        N = ceil(Int, Wstar/ω_smearing)
        if doubleN_flag
            N = round(Int, N/2)
            Neff = 2*N - 1
        else
            Neff = N
        end
        println("ω_smearing provided -> effective number of Chebyshev "*
            "polynomials = $Neff")
        if Nω==nothing
            Nω = round(Int,5*ωmax/ω_smearing)
        end
    elseif N!=nothing && ω_smearing==nothing
        if doubleN_flag
            N = round(Int, N/2)
            Neff = 2*N - 1
        else
            Neff = N
        end
        println("N provided -> effective number of Chebyshev polynomials "*
            "= $Neff")
        if Nω==nothing 
            Nω = round(Int, 5*ωmax*N/Wstar)
        end
    elseif ω_smearing==nothing && N==nothing
        error("ERROR: provide either ω_smearing or N")
    else 
        error("ERROR: provide only ω_smearing or N, not both")
    end
    println()

    # continue checks
    if N < 3
        error("ERROR: N must be >= 2, check ω_smearing or N")
    elseif Nω < 2
        error("ERROR: Nω must be >= 2")
    end

    # ω
    ω = collect(range(0, ωmax, length=Nω))

    # rescaled half-bandwidth W', Eq.(20)
    Wp = 1 - ϵt/2

    # rescaled ω', Eq.(21a)
    a = Wstar/(2*Wp)
    ωp = ω/a .- Wp

    # identity operator
    I = Operators.I_op(sites)

    # rescaled H', Eq.(21b)
    Hp = add(1/a * add(H, -E0*I; cutoff=cutoff), -Wp*I; cutoff=cutoff)

    # Chebyshev vectors
    if Etrunc_flag
        tn = Chebyshev_vectors(C, Hp, ψ0, N; cutoff=cutoff, dK=dK, 
            nsweeps=nsweeps, ϵP=ϵP)
    else
        tn = Chebyshev_vectors(C, Hp, ψ0, N; cutoff=cutoff)
    end

    # Chebyshev moments
    if doubleN_flag==true 
        μn = Chebyshev_moments(ψ0, B, tn; C=C)
    else
        μn = Chebyshev_moments(ψ0, B, tn)
    end

    # number of effective Chebyshev polynomials
    @assert Neff == length(μn)

    # Jackson damping
    gn = Jackson_damping(Neff)

    # Chebyshev polynomials
    Tn = Chebyshev_polynomials(ωp, Neff)

    # χ, Eq.(5)
    prefactor = 1/a * 1 ./ (pi*sqrt.(1 .- ωp.^2))
    sumval = sum([gn[n]*μn[n]*Tn[:, n] for n in 2:Neff])
    χ = prefactor .* (gn[1]*μn[1] .+ 2*sumval)

    return χ
end

""" 
Plot local spectral function χ_i(ω) as heatmap with i as horizontal axis 
    and ω as vertical axis

Notes:
- n_xticks is the number of xticks (default: L)

GC, 16-07-2026
"""
function plot_local_spectral_function_heatmap(χi::Vector{Vector{Float64}}, 
    ωmax::Float64; 
    cbar_label::String="χ", title::String="",
    n_xticks::Int=length(χi),
    save=false,fig_dir_name="/Users/scgo/Downloads/p")

    # build χ[ω,i] matrix
    χmat = reduce(hcat, χi)
    ## dimensions
    Nω, L = size(χmat)

    # ωrange 
    ωrange = range(0, ωmax, length=Nω)

    # irange
    irange = 1:L

    # build xticks
    xt = round.(Int, range(1, L, length=n_xticks))
    xt_labels = [string(x) for x in xt]

    # plot
    fig = heatmap(
        irange,
        ωrange,
        χmat,
        xlabel = L"i",
        ylabel = L"$\omega$",
        colorbar_title = latexstring(cbar_label),
        title = title,
        xticks = (xt, xt_labels)
    )

    # save figure
    if save==true
        savefig(fig_dir_name*".svg")
    end

    return fig
end

# plot local spectral function as lines for selected sites
## GC, 29-04-2026
function plot_local_spectral_function_lines(χi::Vector{Vector{Float64}}, 
    ωmax::Float64; 
    i_selected::Vector{Int} = [1],
    save=false, fig_dir_name="/Users/scgo/Downloads/p_lines")
    #=
        Plot local spectral function χ_i(ω) for each i from i_selected as 
            lineplot with ω as horizontal axis
    =#

    χmat = reduce(hcat, χi)

    Nω, L = size(χmat)
    ωrange = range(0, ωmax, length=Nω)

    # selected sites
    for i in i_selected
        if i<1 || i>L
            error("Site index $i out of bounds (1:$L)")
        end
    end

    fig = plot(
        xlabel = L"$\omega$",
        ylabel = L"$\chi$",
        legend = :best,
        grid = false
    )
    for i in i_selected
        plot!(fig, ωrange, χmat[:, i], label=latexstring("\$i=$i\$"))
    end

    # save
    if save == true
        savefig(fig_dir_name * ".svg")
    end

    return fig
end

""" 
Plot spectral function FT χ_q(ω) as heatmap with q as horizontal axis 
    and ω as vertical axis

Notes:
- n_xticks is the number of xticks (default: 3)

GC, 16-07-2026
"""
function plot_FT_spectral_function_heatmap(χq::Vector{Vector{Float64}}, 
    ωmax::Float64; 
    cbar_label::String="|χ(q,ω)|", title::String="",
    n_xticks::Int=3,
    save=false,fig_dir_name="/Users/scgo/Downloads/p")

    # build χ[ω,q] matrix
    χmat = reduce(hcat, χq)
    ## dimensions
    Nω, Nq = size(χmat)

    # ωrange 
    ωrange = range(0, ωmax, length=Nω)

    # qrange (q/π)
    qrange = 2 .* (0:Nq-1) ./ Nq

    # build xticks
    xt_idx = round.(Int, range(1, Nq, length=n_xticks))
    xt = qrange[xt_idx]
    xt_labels = [
        x == 1 ? L"$0$" : latexstring("\\frac{$(x-1)}{$Nq}")
        for x in xt_idx
    ]

    # plot
    fig = heatmap(
        qrange,
        ωrange,
        χmat,
        xlabel = L"$q/\pi$",
        ylabel = L"$\omega$",
        colorbar_title = latexstring(cbar_label),
        title = title,
        xticks = (xt, xt_labels)
    )

    # save figure
    if save==true
        savefig(fig_dir_name*".svg")
    end

    return fig
end

""" 
Projects out high-energy components of ψ within Krylov approximation (see 
    Section III-B)

Notes:
- in DMRG, 1 sweep is left-to-right + right-to-left; here, it is one or the 
    other

GC, 22-07-2026
"""
function proj_highE(ψ::MPS, H::MPO, dK::Int, nsweeps::Int, ϵP::Float64;
    cutoff::Float64=1e-8)

    # number of sites
    L = length(ψ)

    # sweeps loop
    for sweep in 1:nsweeps
        print(stdout, "sweep = ", sweep, " ")
        flush(stdout)

        ## sites range 
        sites_range = isodd(sweep) ? (1:L) : (L:-1:1)

        ## sites loop
        for i in sites_range

            ### ψ in site canonical form
            orthogonalize!(ψ, i)

            ### build Krylov subspace |k>, Eq.(36a)
            k = Vector{MPS}(undef, dK)
            k[1] = deepcopy(ψ)
            for it in 2:dK
                """
                Without truncation, bond dimensions grow too much.
                For that reason, we use cutoff (consistent with rest of code)
                """
                #k[it] = apply(H, k[it-1], method="naive", truncate=false)
                k[it] = apply(H, k[it-1]; cutoff=cutoff)
            end

            ### orthonormalize via Gram-Schmidt, Eq.(36b)
            k_ortho = Vector{MPS}()
            for it in 1:dK
                ktmp = deepcopy(k[it])
                for k2 in k_ortho
                    ktmp -= inner(k2, ktmp) * k2
                end
                #### discard linearly dependent vectors (norm < 1e-12)
                nrm = norm(ktmp)
                if nrm > 1e-12
                    push!(k_ortho, ktmp/nrm)
                end
            end

            ### H in Krylov subspace, Eq.(36c)
            dK_eff = length(k_ortho)
            HK = zeros(dK_eff, dK_eff)
            for it1 in 1:dK_eff, it2 in 1:dK_eff
                HK[it1,it2] = inner(k_ortho[it1]', H, k_ortho[it2])
            end

            ### diagonalize
            @assert isapprox(HK, HK', atol=1e-12, rtol=0)
            vals, vecs = eigen(Hermitian(HK))

            ### apply projection operator, Eqs.(38,39)
            idx_high = findall(vals .>= ϵP)
            vecs_high = vecs[:, idx_high]
            for alpha in 1:size(vecs_high,2)
                #### construct |ϕ_α> = sum_k vecs[k,α] |k> 
                ϕ = vecs_high[1,alpha] * k_ortho[1]
                for it in 2:dK_eff
                    ϕ += vecs_high[it,alpha] * k_ortho[it]
                end

                #### subtract |ϕ_α> <ϕ_α|ψ>
                ψ -= inner(ϕ,ψ) * ϕ
            end
        end

    end

    return ψ
end

end