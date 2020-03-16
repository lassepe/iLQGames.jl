"""
$(TYPEDEF)

The abstract representation of the cost for a player in a game.
"""
abstract type PlayerCost end

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

"""
$(TYPEDEF)

Represents a parameter free player cost that is defined by a funciton f.
"""
struct FunctionPlayerCost{F} <: PlayerCost
    f::F
end
(pc::FunctionPlayerCost)(args...) = pc.f(args...)

"""
$(TYPEDEF)

Represents the quadratic players costs for a singel player in a game.

# Fields

$(TYPEDFIELDS)
"""
struct QuadraticPlayerCost{nx, nu, TL<:SVector{nx}, TQ<:SMatrix{nx, nx},
                           TRL<:SVector{nu}, TRQ<:SMatrix{nu, nu}} <: PlayerCost
    "The linear state cost"
    l::TL
    "The qudratic state cost matrix"
    Q::TQ
    "The linear term of the input cost."
    r::TRL
    "A square matrix to represent the quadratic control cost for this player"
    R::TRQ
end

(pc::QuadraticPlayerCost)(::Any, x::SVector, u::SVector) = pc(x, u)

function (pc::QuadraticPlayerCost)(x::SVector, u::SVector)
    return 1//2 * x'*pc.Q*x + pc.l'*x + 1//2 * u'*pc.R*u + pc.r'*u
end
