"""
$(TYPEDEF)

The abstract representation of the cost for a player in a game.

# Parameters

- `nx`    the number of states
- `nu`:   the number of controls
"""
abstract type PlayerCost{nx, nu} end

"""
    $(FUNCTIONNAME)(x::SVector{nx}, u::SVector{nu}, t::AbstractFloat)

A functor to evaluate the cost representation.
"""
function (pc::PlayerCost) end

"""
    $FUNCTIONNAME(pc::PlayerCost, x::SVector{nx}, u::SVector{nu}, t::AbstractFloat)

Computes the quadratic approximation of the cost.

We provide a convencience default below.
"""
function quadraticize end

"""
    $(FUNCTIONNAME)(pc::PlayerCost)

Returns the index of the player this cost is associated with.
"""
function player_id end

n_states(::Type{<:PlayerCost{nx}}) where {nx} = nx
n_states(p::PlayerCost) = n_states(typeof(p))
n_controls(::Type{<:PlayerCost{nx, nu}}) where {nx, nu} = nu
n_controls(p::PlayerCost) = n_controls(typeof(p))

"""
$(TYPEDEF)

Represents the quadratic players costs for a singel player in a game.

# Parameters

- `nx`: the number of states
- `nu`: the number of controls

# Fields

$(TYPEDFIELDS)
"""
struct QuadraticPlayerCost{nx, nu, TL<:SVector{nx}, TQ<:SMatrix{nx, nx}, TR<:SMatrix{nu, nu}} <: PlayerCost{nx, nu}
    "The linear state cost"
    l::TL
    "The qudratic state cost matrix"
    Q::TQ
    "A square matrix to represent the quadratic control cost for this player"
    R::TR
end
(pc::QuadraticPlayerCost)(::Any, x::SVector, u::SVector) = pc(x, u)
(pc::QuadraticPlayerCost)(x::SVector, u::SVector) = 1//2 * x'*pc.Q*x + pc.l'*x + 1//2 * u'*pc.R*u
