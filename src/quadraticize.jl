"""
    $FUNCTIONNAME(pc::PlayerCost, x::SVector{nx}, u::SVector{nu}, t::AbstractFloat)

A convencience implementation of the cost quadraticization.
"""
function _quadraticize_ad(pc::PlayerCost, g::GeneralGame, x::SVector, u::SVector,
                          t::AbstractFloat)
    @warn "You are using the fallback quadraticization using ForwardDiff.
    Consider implementing a custom `quadraticize` for your `ControlSystem`
    type." maxlog=1

    x_cost = x->pc(g, x, u, t)
    u_cost = u->pc(g, x, u, t)
    # we can compute the gradient and the hessian in one go
    diff_x = DiffResults.HessianResult(x)
    diff_x = ForwardDiff.hessian!(diff_x, x_cost, x)
    # the linear state component of the cost is the gradient in x
    l = DiffResults.gradient(diff_x)
    # the quadratic state component of the cost is the hessian in x
    Q = DiffResults.hessian(diff_x)

    diff_u = DiffResults.HessianResult(u)
    diff_u = ForwardDiff.hessian!(diff_u, u_cost, u)
    # the linear control cost is the gradient in u
    r = DiffResults.gradient(diff_u)
    # the quadratic control component o the cost is the hessian in u
    R = DiffResults.hessian(diff_u)

    return QuadraticPlayerCost(l, Q, r, R)
end

quadraticize!(qcache::QuadCache, pc::PlayerCost, g::GeneralGame, x::SVector,
              u::SVector, t::AbstractFloat) = _quadraticize_ad(pc, g, x, u, t)
