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


"--------------------- Convencience Implementations ---------------------"

"""
    $FUNCTIONNAME(pc::PlayerCost, x::SVector{nx}, u::SVector{nu}, t::AbstractFloat)

A convencience implementation of the cost quadraticization.
"""
function quadraticize(pc::PlayerCost{nx, nu}, x::SVector{nx}, u::SVector{nu}, t::AbstractFloat) where {nx, nu}
    # the linear state component of the cost is the gradient in x
    l = ForwardDiff.gradient(x->pc(x, u, t), x)
    # the quadratic state component of the cost is the hessian in x
    Q = ForwardDiff.hessian(x->pc(x, u, t), x)
    # the quadratic control component o the cost is the hessian in u
    R = ForwardDiff.hessian(u->pc(x, u, t), u)

    return QuadraticPlayerCost(Q, l, R)
end

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
(pc::QuadraticPlayerCost)(x::SVector, u::SVector) = x'*pc.Q*x + l'*pc.x + u'*pc.R*u
