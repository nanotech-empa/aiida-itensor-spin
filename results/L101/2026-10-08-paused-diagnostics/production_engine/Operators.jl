module Operators

# packages #
using ITensorMPS
###############################################################################

# S^2 operator
## GC, 22-04-2026
function S2_op(sites)
    #=
		S^2 operator

        S^2 = SpSm + Sz^2 - Sz
        SpSm = sum_{i,j} Sp(i)Sm(j)
        Sz^2 = sum_{i,j} Sz(i)Sz(j)
        Sz = sum_i Sz(i)
	=#

    ampo = AutoMPO()
    
    # Nsites
    Nsites = length(sites)

    for i in 1:Nsites
        for j in 1:Nsites
            ampo += 1.,"S+",i,"S-",j
            ampo += 1.,"Sz",i,"Sz",j
        end
        ampo += -1.,"Sz",i
    end

    S2 = MPO(ampo,sites)

    return(S2)
end

# Sz(i) operator
## GC, 17-01-2022
function Szi_op(i,sites)
    #=
		Sz(i) operator
	=#

    ampo = AutoMPO()

    ampo += 1.,"Sz",i

    Szi = MPO(ampo,sites)

    return(Szi)
end

# Sz operator
## GC, 22-04-2026
function Sz_op(sites)
    #=
		Sz operator

        Sz = sum_i Sz(i)
	=#

    ampo = AutoMPO()

    # Nsites
    Nsites = length(sites)

    for i in 1:Nsites
        ampo += 1.,"Sz",i
    end

    Sz = MPO(ampo,sites)

    return(Sz)
end

# Sx(i) operator
## GC, 13-12-2022
function Sxi_op(i,sites)
    #=
		Sx(i) operator

        Sx(i) = 1/2*[Sp(i) + Sm(i)]
	=#

    ampo = AutoMPO()

    ampo += 0.5,"S+",i
    ampo += 0.5,"S-",i

    Sxi = MPO(ampo,sites)

    return(Sxi)
end

# I*Sy(i) operator
## GC, 13-12-2022
function ISyi_op(i,sites)
    #=
        I*Sy(i) operator

        I*Sy(i) = 1/2*[Sp(i) - Sm(i)]
	=#

    ampo = AutoMPO()

    ampo += 0.5,"S+",i
    ampo += -0.5,"S-",i

    ISyi = MPO(ampo,sites)

    return(ISyi)
end

# n(i) operator
## GC, 26-04-2024
function ni_op(i,sites)
    #=
		n(i) operator
	=#

    ampo = AutoMPO()

    ampo += 1.,"Ntot",i

    ni = MPO(ampo,sites)

    return(ni)
end

# S(i)S(j) operator
## GC, 19-03-2026
function SiSj_op(i,j,sites)
    #=
		vec{S}(i) . vec{S}(j) operator
	=#
    ampo = AutoMPO()

    ampo += 1.,"Sz",i,"Sz",j
    ampo += 0.5,"S+",i,"S-",j
    ampo += 0.5,"S-",i,"S+",j

    SiSj = MPO(ampo,sites)

    return SiSj
end

# identity operator
## GC, 23-04-2026
function I_op(sites)
    #=
		identity operator
	=#
    I = MPO(sites, "Id")

    return I
end

# S+(i) operator
## GC, 24-04-2026
function Spi_op(i,sites)
    #=
		S+(i) operator
	=#

    ampo = AutoMPO()

    ampo += 1.,"S+",i

    Spi = MPO(ampo,sites)

    return(Spi)
end

# S-(i) operator
## GC, 24-04-2026
function Smi_op(i,sites)
    #=
		S-(i) operator
	=#

    ampo = AutoMPO()

    ampo += 1.,"S-",i

    Smi = MPO(ampo,sites)

    return(Smi)
end

# Sz(i)Sz(j) operator
## GC, 29-04-2026
function SziSzj_op(i,j,sites)
    #=
		Sz(i) Sz(j) operator
	=#
    ampo = AutoMPO()

    ampo += 1.,"Sz",i,"Sz",j

    SziSzj = MPO(ampo,sites)

    return SziSzj
end

end