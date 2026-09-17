module Hamiltonians

# packages #
using ITensorMPS
###############################################################################

# open-ended D1 4-site Hubbard chain
## GC, 22-04-2026
function H_4siteD1_1D_Hubb_OBC(Nfst,t,tt,tp,U,sites;pS2=0.0)
    #=
		open-ended 4-site Hubbard chain where unit cell has D1 symmetry

		(1)--(3)--(4)....(5)
		      |
			  |
			  |
			 (2)

	    -- , -t
		— — — , -tt
		.... , -tp

        Notes:
        - if tt=t, unit cell has C3 symmetry
        - p*S^2 (together with conservation of Sz) may be used to select states
            with given S
	=#

    if length(sites) != 4*Nfst
        println("ERROR: inconsistent number of sites")
        exit()
    end

    ampo = AutoMPO()

    # t
    nbt = [-2,+1]
    for unit in 1:Nfst, j in nbt
        ampo += -t,"Cdagup",4*unit-1,"Cup",4*unit-1+j #up
        ampo += -t,"Cdagdn",4*unit-1,"Cdn",4*unit-1+j #dn
        #hc
        ampo += -t,"Cdagup",4*unit-1+j,"Cup",4*unit-1
        ampo += -t,"Cdagdn",4*unit-1+j,"Cdn",4*unit-1
    end

    # tt
    for unit in 1:Nfst
        ampo += -tt,"Cdagup",4*unit-1,"Cup",4*unit-2 #up
        ampo += -tt,"Cdagdn",4*unit-1,"Cdn",4*unit-2 #dn
        #hc
        ampo += -tt,"Cdagup",4*unit-2,"Cup",4*unit-1
        ampo += -tt,"Cdagdn",4*unit-2,"Cdn",4*unit-1
    end

    # tp
    for unit in 1:Nfst-1
        ampo += -tp,"Cdagup",4*unit,"Cup",4*unit+1 #up
        ampo += -tp,"Cdagdn",4*unit,"Cdn",4*unit+1 #dn
        #hc
        ampo += -tp,"Cdagup",4*unit+1,"Cup",4*unit
        ampo += -tp,"Cdagdn",4*unit+1,"Cdn",4*unit
    end

    # U
	for i in 1:4*Nfst
        ampo += U,"Nupdn",i
	end

	# p*S^2
    if pS2 != 0.0
    	for i in 1:4*Nfst
            for j in 1:4*Nfst
                ampo += pS2,"S+",i,"S-",j
                ampo += pS2,"Sz",i,"Sz",j
            end
            ampo += -pS2,"Sz",i
        end
    end

    H = MPO(ampo,sites)

    return(H)
end

# open-ended bilinear biquadratic Heisenberg chain
## GC, 22-04-2026
function H_Heis_BLBQ_1D_OBC(N,J,beta,sites;pS2=0.0)
    #=
        open-ended bilinear biquadratic Heisenberg chain

        (1)--(2)

        --, J*vec{S}.vec{S}
            J*beta*(vec{S}.vec{S})^2

        Notes:
        - if beta=0, Heisenberg chain is recovered
    =#

    if length(sites) != N
        println("ERROR: inconsistent number of sites")
        exit()
    end

    ampo = AutoMPO()

    # J
    for i in 1:N-1
        ampo += J,"Sz",i,"Sz",i+1
        ampo += J/2,"S+",i,"S-",i+1
        ampo += J/2,"S-",i,"S+",i+1
    end

    # beta
    if beta != 0.0
        for i in 1:N-1
            ampo += J*beta,"Sz*Sz",i,"Sz*Sz",i+1
            ampo += J*beta/2,"Sz*S+",i,"Sz*S-",i+1
            ampo += J*beta/2,"Sz*S-",i,"Sz*S+",i+1
            ampo += J*beta/2,"S+*Sz",i,"S-*Sz",i+1
            ampo += J*beta/2,"S-*Sz",i,"S+*Sz",i+1
            ampo += J*beta/4,"S+*S+",i,"S-*S-",i+1
            ampo += J*beta/4,"S+*S-",i,"S-*S+",i+1
            ampo += J*beta/4,"S-*S+",i,"S+*S-",i+1
            ampo += J*beta/4,"S-*S-",i,"S+*S+",i+1
            #=
            ampo += J*beta,"Sz",i,"Sz",i,"Sz",i+1,"Sz",i+1
            ampo += J*beta/2,"Sz",i,"S+",i,"Sz",i+1,"S-",i+1
            ampo += J*beta/2,"Sz",i,"S-",i,"Sz",i+1,"S+",i+1
            ampo += J*beta/2,"S+",i,"Sz",i,"S-",i+1,"Sz",i+1
            ampo += J*beta/2,"S-",i,"Sz",i,"S+",i+1,"Sz",i+1
            ampo += J*beta/4,"S+",i,"S+",i,"S-",i+1,"S-",i+1
            ampo += J*beta/4,"S+",i,"S-",i,"S-",i+1,"S+",i+1
            ampo += J*beta/4,"S-",i,"S+",i,"S+",i+1,"S-",i+1
            ampo += J*beta/4,"S-",i,"S-",i,"S+",i+1,"S+",i+1
            =#
        end
    end

    # p*S^2
    if pS2 != 0.0
    	for i in 1:N
            for j in 1:N
                ampo += pS2,"S+",i,"S-",j
                ampo += pS2,"Sz",i,"Sz",j
            end
            ampo += -pS2,"Sz",i
        end
    end

    H = MPO(ampo,sites)

    return H
end

# open-ended Hubbard chain
## GC, 22-04-2026
function H_Hubb_1D_OBC(Nsites,t,U,sites;pS2=0.0)
    #=
		open-ended Hubbard chain

		(1)--(2)

	    -- , -t
	=#

    if length(sites) != Nsites
        println("ERROR: inconsistent number of sites")
        exit()
    end

    ampo = AutoMPO()

    # t
    for i in 1:Nsites-1
        ampo += -t,"Cdagup",i,"Cup",i+1 #up
        ampo += -t,"Cdagdn",i,"Cdn",i+1 #dn
        #hc
        ampo += -t,"Cdagup",i+1,"Cup",i
        ampo += -t,"Cdagdn",i+1,"Cdn",i
    end

    # U
	for i in 1:Nsites
        ampo += U,"Nupdn",i
	end

	# p*S^2
    if pS2 != 0.0
    	for i in 1:Nsites
            for j in 1:Nsites
                ampo += pS2,"S+",i,"S-",j
                ampo += pS2,"Sz",i,"Sz",j
            end
            ampo += -pS2,"Sz",i
        end
    end

    H = MPO(ampo,sites)

    return(H)
end

# open-ended J-Jp Heisenberg chain
## GC, 22-04-2026
function H_Heis_JJp_1D_OBC(N,J,Jp,sites;pS2=0.0,hSz=0.0)
    #=
        open-ended J-Jp Heisenberg chain

        (1)--(2)...(3)

        --, J*vec{S}.vec{S}
        ..., Jp*vec{S}.vec{S}

        Notes:
        - if J=Jp, Heisenberg chain is recovered
    =#

    if length(sites) != N
        println("ERROR: inconsistent number of sites")
        exit()
    end

    ampo = AutoMPO()

    #J
    for i in 1:2:N-1
        ampo += J,"Sz",i,"Sz",i+1
        ampo += J/2,"S+",i,"S-",i+1
        ampo += J/2,"S-",i,"S+",i+1
    end

    #Jp
    for i in 2:2:N-1
        ampo += Jp,"Sz",i,"Sz",i+1
        ampo += Jp/2,"S+",i,"S-",i+1
        ampo += Jp/2,"S-",i,"S+",i+1
    end

    # p*S^2
    if pS2 != 0.0
    	for i in 1:N
            for j in 1:N
                ampo += pS2,"S+",i,"S-",j
                ampo += pS2,"Sz",i,"Sz",j
            end
            ampo += -pS2,"Sz",i
        end
    end

    # h*Sz
    if hSz != 0.0
    	for i in 1:N
            ampo += hSz,"Sz",i
        end
    end

    H = MPO(ampo,sites)

    return H
end

"""
Open-ended Heisenberg chain with spin impurity

(1)- ... -(i_imp-1)--(i_imp)---(i_imp+1)- ...

- , J 
-- , JL 
--- , JR 
i_imp = site of spin impurity

GC, 13-07-2026
"""
function H_Heis_impurity_1D_OBC(
    J::Float64, JL::Float64, JR::Float64, i_imp::Int, sites;
    pS2::Float64=0.0)::MPO

    # initialize MPO
    ampo = AutoMPO()

    # chain length
    L = length(sites)

    # exchange couplings
    for i in 1:L-1
        ## JL 
        if i == i_imp-1
            ampo += JL,"Sz",i,"Sz",i+1
            ampo += JL/2,"S+",i,"S-",i+1
            ampo += JL/2,"S-",i,"S+",i+1
        ## JR
        elseif i == i_imp 
            ampo += JR,"Sz",i,"Sz",i+1
            ampo += JR/2,"S+",i,"S-",i+1
            ampo += JR/2,"S-",i,"S+",i+1
        ## J
        else
            ampo += J,"Sz",i,"Sz",i+1
            ampo += J/2,"S+",i,"S-",i+1
            ampo += J/2,"S-",i,"S+",i+1
        end
    end

    # p*S^2 term
    if pS2 != 0.0
    	for i in 1:L
            for j in 1:L
                ampo += pS2,"S+",i,"S-",j
                ampo += pS2,"Sz",i,"Sz",j
            end
            ampo += -pS2,"Sz",i
        end
    end

    # Hamiltonian MPO
    H = MPO(ampo,sites)

    return H
end

"""
Open-ended Heisenberg chain

(1)-(2)- ...

- , J 

GC, 15-07-2026
"""
function H_Heis_1D_OBC(
    J::Float64, sites;
    pS2::Float64=0.0)::MPO

    # initialize MPO
    ampo = AutoMPO()

    # chain length
    L = length(sites)

    # J
    for i in 1:L-1
        ampo += J,"Sz",i,"Sz",i+1
        ampo += J/2,"S+",i,"S-",i+1
        ampo += J/2,"S-",i,"S+",i+1
    end

    # p*S^2 term
    if pS2 != 0.0
    	for i in 1:L
            for j in 1:L
                ampo += pS2,"S+",i,"S-",j
                ampo += pS2,"Sz",i,"Sz",j
            end
            ampo += -pS2,"Sz",i
        end
    end

    # Hamiltonian MPO
    H = MPO(ampo,sites)

    return H
end

end
