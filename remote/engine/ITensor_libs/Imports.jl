lib_dir = @__DIR__
engine_dir = dirname(lib_dir)

include(joinpath(engine_dir, "GenericSpinSites.jl"))
using .GenericSpinSites

include(joinpath(lib_dir, "AuxFunctions.jl"))
include(joinpath(lib_dir, "DMRGsweeps.jl"))
include(joinpath(lib_dir, "Hamiltonians.jl"))
include(joinpath(lib_dir, "Operators.jl"))
include(joinpath(lib_dir, "SpectralFunction.jl"))

import .AuxFunctions
import .DMRGsweeps
import .Hamiltonians
import .Operators
import .SpectralFunction