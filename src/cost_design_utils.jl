struct QuadCost{TQ<:SMatrix}
    Q::TQ
end
@inline (c::QuadCost)(x) = 1/2*x'*c.Q*x
@inline function quad!(Q::MMatrix, l::MVector, c::QuadCost,
                       x::SVector{n}, xi::SVector{n}) where {n}
    Q[xi, xi] += c.Q
    l[xi] += c.Q * x
    return nothing
end

# Interface to implement NPlayerNavigationCost
@with_kw struct SoftConstr
    "The index of the state/input element this applies to."
    id::Int
    "The soft lower bound."
    min::Float64
    "The soft upper bound."
    max::Float64
    "The weight of the soft constraint."
    w::Float64
end
@inline function softconstr(val::Real, min::Real, max::Real, w::Real)
    @assert min < max
    gap = val < min ? val - min : val > max ? val - max : zero(w)
    return w*gap^2
end
@inline function (constr::SoftConstr)(x, xi)
    @unpack id, min, max, w = constr
    val = x[xi[id]]
    return softconstr(val, min, max, w)
end

@inline function quad!(Q::MMatrix, l::MVector, constr::SoftConstr, x::SVector, xi)
    @unpack id, w, min, max = constr
    idx = xi[id]

    @assert min < max
    if x[idx] < min
        Q[idx, idx] += 2w
        l[idx] += 2w*(x[idx] - min)
    elseif x[idx] > max
        Q[idx, idx] += 2w
        l[idx] += 2w*(x[idx] - max)
    end
    return nothing
end

@with_kw struct ProximityCost{TR,TW}
    "A collection of avoidance raddii this player want's to keep from others."
    rs::TR
    "A collection of costs this player has for colliding with others."
    ws::TW
end
ProximityCost(r, w, np) = ProximityCost(SVector{np}([r for _ in 1:np]),
                                        SVector{np}([w for _ in 1:np]))

"Proxmity cost for player i to player j. j is the index of the *other* player."
@inline function (pc::ProximityCost)(xp_i::SVector, xp_j::SVector, j::Int)
    @unpack rs, ws = pc
    Δxp = xp_i - xp_j
    return softconstr(sqrt(Δxp'*Δxp), rs[j], Inf, ws[j])
end

@inline function quad!(Q::MMatrix, l::MVector, pc::ProximityCost, x::SVector,
                       xyi_1, xyi_2, j::Int)
    @unpack rs, ws = pc
    r_avoid = rs[j]
    w = ws[j]

    x1, y1 = xyi_1[1], xyi_1[2]
    x2, y2 = xyi_2[1], xyi_2[2]
    Δx = x[x1] - x[x2]
    Δy = x[y1] - x[y2]
    Δnormsq = Δx^2 + Δy^2
    Δnorm = sqrt(Δnormsq)
    if Δnorm < r_avoid
        # cost model: w*(Δnorm - min)^2
        δx = 2*w*Δx*(Δnorm - r_avoid)/Δnorm
        δy = 2*w*Δy*(Δnorm - r_avoid)/Δnorm
        l[x1] += δx
        l[y1] += δy
        l[x2] -= δx
        l[y2] -= δy

        w̃ = 2*w/Δnormsq

        δxδx = (w̃*Δx^2) + (w̃*Δx^2*(r_avoid - Δnorm))/Δnorm - (w̃*Δnorm*(r_avoid - Δnorm))
        δyδy = (w̃*Δy^2) + (w̃*Δy^2*(r_avoid - Δnorm))/Δnorm - (w̃*Δnorm*(r_avoid - Δnorm))
        δxδy = (w̃*Δx*Δy) + (w̃*Δx*Δy*(r_avoid - Δnorm))/Δnorm

        Q[x1,x1] += δxδx
        Q[x1,x2] -= δxδx; Q[x2,x1] -= δxδx
        Q[x2,x2] += δxδx

        Q[y1,y1] += δyδy
        Q[y1,y2] -= δyδy; Q[y2,y1] -= δyδy
        Q[y2,y2] += δyδy

        Q[x1,y1] += δxδy; Q[y1,x1] += δxδy
        Q[x1,y2] -= δxδy; Q[y2,x1] -= δxδy
        Q[y1,x2] -= δxδy; Q[x2,y1] -= δxδy
        Q[x2,y2] += δxδy; Q[y2,x2] += δxδy
    end
end

@with_kw struct GoalCost{TG<:SVector, TQg<:SMatrix}
    t_active::Float64
    xg::TG
    Qg::TQg
end

@inline function (gc::GoalCost)(x, t)
    @unpack t_active, xg, Qg = gc
    if t >= t_active
        Δx = x - xg
        return 1/2 * Δx'*Qg*Δx
    else
        return 0.0
    end
end

@inline function quad!(Q::MMatrix, l::MVector, gc::GoalCost, x::SVector,
                       xid::SVector{n}, t::AbstractFloat) where {n}
    @unpack t_active, xg, Qg = gc
    if t >= t_active
        Q[xid, xid] += Qg
        l[xid] += Qg*(x-xg)
    end
    return nothing
end
