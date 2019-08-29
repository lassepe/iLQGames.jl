"""
$(TYPEDEF)

The abstract representation of the cost for a player in a game.

# Parameters

- `nx`    the number of states
- `nu`:   the number of controls
"""
abstract type PlayerCost{nx, nu} end

"""
    $(FUNCTIONNAME)(x::SVector{nx}, u::SVector{nu}, t::Real)

A functor to evaluate the cost representation.
"""
function (pc::PlayerCost) end

"""
    $FUNCTIONNAME(pc::PlayerCost, x::SVector{nx}, u::SVector{nu}, t::Real)

Computes the quadratic approximation of the cost.

We provide a convencience default below.
"""
function quadraticize end


"--------------------- Convencience Impelemtnations ---------------------"


"""
    $FUNCTIONNAME(pc::PlayerCost, x::SVector{nx}, u::SVector{nu}, t::Real)

A convencience implementaiton of the cost quadraticization.
"""
function quadraticize(pc::PlayerCost{nx, nu}, x::SVector{nx}, u::SVector{nu}, t::Real) where {nx, nu}
    # the linear state component of the cost is the gradient in x
    l = ForwardDiff.gradient(x->pc(x, u, t), x)
    # the quadratic state component of the cost is the hessian in x
    Q = ForwardDiff.hessian(x->pc(x, u, t), x)
    # the quadratic control component o the cost is the hessian in u
    R = ForwardDiff.hessian(u->pc(x, u, t), u)
end
