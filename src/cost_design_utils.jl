@inline function proximitycost(xp1::SVector{n}, xp2::SVector{n},
                               r_avoid::AbstractFloat, w::AbstractFloat) where {n}
    Δxp = xp1 - xp2
    return softconstr(sqrt(Δxp'*Δxp), r_avoid, Inf, w)
end

@inline function proximitycost_quad!(Q::MMatrix, l::MVector, x::SVector,
                                 r_avoid::Real, w::Real, x1::Int, y1::Int,
                                 x2::Int, y2::Int)
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

@inline function softconstr(val::Real, min::Real, max::Real, w::Real)
    @assert min < max
    gap = val < min ? val - min : val > max ? val - max : zero(w)
    return w*gap^2
end

@inline function softconstr_quad!(Q::MMatrix, l::MVector, x::SVector,
                                  min::Real, max::Real, w::Real, idx::Int)
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

@inline function softconstr_quad!(Q::MMatrix, x::SVector, min::Real, max::Real,
                                  w::Real, idx::Int)
    @assert min < max
    if !(min < x[idx] < max)
        Q[idx, idx] += 2w
    end
    return nothing
end

@inline function goalstatecost(Qg::SMatrix{n,n}, xg::SVector{n}, x::SVector{n},
                               t::AbstractFloat, t_active::AbstractFloat) where {n}
    if t >= t_active
        Δx = x - xg
        return 1/2 * Δx'*Qg*Δx
    else
        return 0.0
    end
end

@inline function goalstatecost_quad!(Q::MMatrix, l::MVector, Qg::SMatrix{n, n},
                                     xg::SVector{n}, x::SVector,
                                     xid::SVector{n}, t::AbstractFloat,
                                     t_active::AbstractFloat) where {n}
    if t >= t_active
        Q[xid, xid] += Qg
        l[xid] += Qg*(x-xg)
    end
    return nothing
end

@inline statecost(Qs::SMatrix{n,n}, x::SVector{n}) where {n} = 1/2*x'*Qs*x
@inline function statecost_quad!(Q::MMatrix, l::MVector, Qs::SMatrix{n,n},
                                 x::SVector{n}, xid::SVector{n}) where {n}
    Q[xid, xid] += Qs
    l[xid] += Qs * x
    return nothing
end

@inline inputcost(R::SMatrix{n,n}, u::SVector{n}) where {n} = 1/2*u'*R*u
@inline function inputcost_quad!(R::MMatrix, Rᵢ::SMatrix{n,n}, uid::SVector{n}) where {n}
    R[uid, uid] += Rᵢ
    return nothing
end

