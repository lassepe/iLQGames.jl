using DocStringExtensions
using StaticArrays

"""
$(TYPEDEF)

Represents the quadratic players costs for a singel player in a game.

# Parameters

- `nx`: the number of states
- `nu`: the number of controls

# Fields

$(TYPEDFIELDS)
"""
struct QuadraticPlayerCost{nx, nu, TQ<:SMatrix{nx, nx}, TL<:SVector{nx}, TR<:SMatrix{nu, nu}}
    "The qudratic state cost matrix"
    Q::TQ
    "The linear state cost"
    l::TL
    "A square matrix to represent the quadratic control cost for this player"
    R::TR
end
n_states(qpc::QuadraticPlayerCost{nx}) where {nx} = nx
n_controls(qpc::QuadraticPlayerCost{nx, nu}) where {nx, nu} = nu
