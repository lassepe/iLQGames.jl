"""
    $FUNCTIONNAME(pc::PlayerCost, x::SVector{nx}, u::SVector{nu}, t::AbstractFloat)

A convencience implementation of the cost quadraticization.
"""
function _quadraticize_ad(pc::PlayerCost{nx, nu}, g::GeneralGame, x::SVector{nx},
                          u::SVector{nu}, t::AbstractFloat) where {nx, nu}
    @warn "You are using the fallback quadraticization using ForwardDiff.
    Consider implementing a custom `quadraticize` for your `ControlSystem`
    type." maxlog=1

    x_cost = x->pc(g, x, u, t)
    u_cost = u->pc(g, x, u, t)
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
quadraticize(pc::PlayerCost, g::GeneralGame, x::SVector, u::SVector,
             t::AbstractFloat) = _quadraticize_ad(pc, g, x, u, t)
