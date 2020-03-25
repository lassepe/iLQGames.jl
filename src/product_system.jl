"Computes the joint state and control dimensions of the combined system."
function xu_dims(subsystems::NTuple)
    nx, nu = 0, 0;
    xids, uids = [], []
    for sub in subsystems
        nxᵢ = n_states(sub); nuᵢ = n_controls(sub)
        push!(xids, SVector{nxᵢ}((nx+1):(nx+nxᵢ)))
        push!(uids, SVector{nuᵢ}((nu+1):(nu+nuᵢ)))
        nx += nxᵢ; nu += nuᵢ
    end
    xyids = (xid[xyindex(sub)] for (xid, sub) in zip(xids, subsystems))
    return nx, nu, Tuple(xids), Tuple(uids), Tuple(xyids)
end

function ξ_dims(subsystems::NTuple)
    nξ = 0
    ξids = []
    for sub in subsystems
        nξᵢ = n_linstates(sub)
        push!(ξids, SVector{nξᵢ}((nξ+1):(nξ+nξᵢ)))
        nξ += nξᵢ
    end
    ξxyids = (ξid[ξxyindex(sub)] for (ξid, sub) in zip(ξids, subsystems))
    return nξ, Tuple(ξids), Tuple(ξxyids)
end

Base.@pure function common_linstyle(subsystems::NTuple)
    linstyles = unique(LinearizationStyle.(subsystems))
    return length(linstyles) == 1 ? linstyles[1] : return JacobianLinearization()
end

struct ProductSystem{ΔT,nx,nu,np,nξ,TS<:NTuple{np,<:ControlSystem{ΔT}},TXI<:NTuple{np},
                     TUI<:NTuple{np},TXY<:NTuple{np},TXII<:NTuple{np},TXIXY<:NTuple{np}}<:ControlSystem{ΔT,nx,nu}
    subsystems::TS
    xids::TXI
    uids::TUI
    xyids::TXY
    ξids::TXII
    ξxyids::TXIXY
    function ProductSystem(subsystems::TS) where {ΔT,np,TS<:NTuple{np,<:ControlSystem{ΔT}}}
        nx, nu, xids, uids, xyids = xu_dims(subsystems)
        # Notice: this very local type instability is acceptable because the
        # ProductSystem will only be construted once
        if common_linstyle(subsystems) isa FeedbackLinearization
            nξ, ξids, ξxyids = ξ_dims(subsystems)
        else
            nξ, ξids, ξxyids = 0, Tuple(0 for i in 1:np), Tuple(0 for i in 1:np)
        end
        new{ΔT,nx,nu,np,nξ,TS,typeof(xids),typeof(uids),
            typeof(xyids),typeof(ξids),typeof(ξxyids)}(subsystems, xids, uids,
                                                       xyids, ξids, ξxyids)
    end
end
n_players(cs::ProductSystem{ΔT,nx,nu,np}) where {ΔT,nx,nu,np} = np
subsystems(cs::ProductSystem) = cs.subsystems
xindex(cs::ProductSystem) = cs.xids
uindex(cs::ProductSystem) = cs.uids
xyindex(cs::ProductSystem) = cs.xyids

function dx(cs::ProductSystem, x::SVector, u::SVector, t::AbstractFloat)
    dxs = map(subsystems(cs), xindex(cs), uindex(cs)) do sub, xid, uid
        xᵢ = x[xid]
        uᵢ = u[uid]
        return dx(sub, xᵢ, uᵢ, t)
    end
    return vcat(dxs...)
end

# TODO: maybe make this a generated funtions. Depending on the dimensionality,
# it might be benificial to do the sparse or dense `next_x`.
function next_x(cs::ProductSystem, x::SVector, u::SVector, t::Float64)
    xs = map(subsystems(cs), xindex(cs), uindex(cs)) do sub, xid, uid
        xᵢ = x[xid]
        uᵢ = u[uid]
        return next_x(sub, xᵢ, uᵢ, t)
    end
    return vcat(xs...)
end

# computing large matrix exponentials is expensive. Therefore, we exploit the
# sparsity of the dynamics and compute the linearization for each subsystems
@inline function linearize_discrete(cs::ProductSystem, x::SVector, u::SVector, t::AbstractFloat)
    nx = n_states(cs)
    nu = n_controls(cs)
    # the full matrices
    A = @MMatrix zeros(nx, nx)
    B = @MMatrix zeros(nx, nu)
    for (i, sub) in enumerate(subsystems(cs))
        xid = xindex(cs)[i]
        uid = uindex(cs)[i]

        lin_sub = linearize_discrete(sub, x[xid], u[uid], t)
        A[xid, xid] = lin_sub.A
        B[xid, uid] = lin_sub.B
    end
    return LinearSystem{samplingtime(cs)}(SMatrix(A), SMatrix(B))
end

"------------------- Implement Feedback Linearization Interface -------------------"

# For now, the product system is only feedback linearizable if all subsystems are
LinearizationStyle(cs::ProductSystem) = common_linstyle(subsystems(cs))

n_linstates(cs::ProductSystem{ΔT,nx,nu,np,nξ}) where {ΔT,nx,nu,np,nξ} = nξ
ξxyindex(cs::ProductSystem) = cs.ξxyids
ξindex(cs::ProductSystem) = cs.ξids

function feedbacklin(cs::ProductSystem)
    nx = n_states(cs)
    nu = n_controls(cs)
    # the full matrices
    A = @MMatrix zeros(nx, nx)
    B = @MMatrix zeros(nx, nu)
    for (i, sub) in enumerate(subsystems(cs))
        xid = xindex(cs)[i]
        uid = uindex(cs)[i]

        lin_sub = feedbacklin(sub)
        A[xid, xid] = lin_sub.dyn.A
        B[xid, uid] = lin_sub.dyn.B
    end
    dyn = LinearSystem{samplingtime(cs)}(SMatrix(A), SMatrix(B))
    return LTISystem(dyn, ξxyindex(cs), ξindex(cs))
end

"A sparse version of the `feedbacklin` method that may be faster if the product
system contains of many subsystems (exploits block diagonal structure of linearized
system matrix)."
function sparse_feedbacklin(cs::ProductSystem)
    feedbacklin_subs = map(subsystems(cs)) do sub
        feedbacklin(sub)
    end
    return ProductSystem(feedbacklin_subs)
end

function x_from(cs::ProductSystem, ξ::SVector)
    subs = subsystems(cs)
    ξids = ξindex(cs)
    xs = map(pindex(cs)) do i
        x_from(subs[i], ξ[ξids[i]])
    end
    # Notice: we can safely vcat because in a product system, xids will always be
    # ordered in an ascending order.
    return vcat(Tuple(xs)...)
end

function ξ_from(cs::ProductSystem, x::SVector)
    subs = subsystems(cs)
    xids = xindex(cs)
    ξs = map(pindex(cs)) do i
        ξ_from(subs[i], x[xids[i]])
    end
    # Notice: we can safely vcat because in a product system, ξids will always be
    # ordered in an ascending order.
    return vcat(Tuple(ξs)...)
end

function λ_issingular(cs::ProductSystem, ξ::SVector)
    return any(zip(subsystems(cs), ξindex(cs))) do (subᵢ, ξis)
        λ_issingular(subᵢ, ξ[ξis])
    end
end

# TODO: why does this allocate?
function inverse_decoupling_matrix(cs::ProductSystem, x::SVector)
    nu = n_controls(cs)
    Minv = @MMatrix(zeros(nu, nu))
    for (sub, xid, uid) in zip(subsystems(cs), xindex(cs), uindex(cs))
        xᵢ = x[xid]
        Minv[uid, uid] = inverse_decoupling_matrix(sub, xᵢ)
    end
    return SMatrix(Minv)
end

function decoupling_drift_term(cs::ProductSystem, x)
    nu = n_controls(cs)
    subs = subsystems(cs)
    xids = xindex(cs)
    uids = uindex(cs)

    ms = map(pindex(cs)) do i
        decoupling_drift_term(subs[i], x[xids[i]])
    end
    # Notice: we can safely vcat because in a product system, uids will always be
    # ordered in an ascending order.
    return vcat(Tuple(ms)...)
end

function transformed_cost(cs::ProductSystem, c::PlayerCost, np::Int)
    # forward to the corresponding subsystem
    return transformed_cost(subsystems(cs)[player_id(c)], c, np)
end
