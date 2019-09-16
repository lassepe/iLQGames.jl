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

n_states(::Type{<:PlayerCost{nx}}) where {nx} = nx
n_states(p::PlayerCost) = n_states(typeof(p))
n_controls(::Type{<:PlayerCost{nx, nu}}) where {nx, nu} = nu
n_controls(p::PlayerCost) = n_controls(typeof(p))


"--------------------- Convencience Implementations ---------------------"

"""
    $FUNCTIONNAME(pc::PlayerCost, x::SVector{nx}, u::SVector{nu}, t::AbstractFloat)

A convencience implementation of the cost quadraticization.
"""
function quadraticize(pc::PlayerCost{nx, nu}, x::SVector{nx}, u::SVector{nu}, t::AbstractFloat) where {nx, nu}
    x_cost = x->pc(x, u, t)
    u_cost = u->pc(x, u, t)
    # for the state cost we can compute the gradient and the hessian in one go
    diff_x = DiffResults.HessianResult(x)
    diff_x = ForwardDiff.hessian!(diff_x, x_cost, x)
    # the linear state component of the cost is the gradient in x
    l = DiffResults.gradient(diff_x)
    # the quadratic state component of the cost is the hessian in x
    Q = DiffResults.hessian(diff_x)
    # the quadratic control component o the cost is the hessian in u
    R = ForwardDiff.hessian(u_cost, u)

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
struct QuadraticPlayerCost{nx, nu, TQ<:SMatrix{nx, nx}, TL<:SVector{nx}, TR<:SMatrix{nu, nu}} <: PlayerCost{nx, nu}
    "The qudratic state cost matrix"
    Q::TQ
    "The linear state cost"
    l::TL
    "A square matrix to represent the quadratic control cost for this player"
    R::TR
end
(pc::QuadraticPlayerCost)(x::SVector, u::SVector) = 1//2 * x'*pc.Q*x + pc.l'*x + 1//2 * u'*pc.R*u
