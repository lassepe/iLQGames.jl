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
    return length(linstyles) == 1 ? linstyles[1] : return DefaultLinearization()
end

struct ProductSystem{ΔT,nx,nu,np,nξ,TS<:NTuple{np,<:ControlSystem{ΔT}},TXI<:NTuple{np},
                     TUI<:NTuple{np},TXY<:NTuple{np},TXII,TXIXY}<:ControlSystem{ΔT,nx,nu}
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
            nξ, ξids, ξxyids = nothing, nothing, nothing
        end
        new{ΔT,nx,nu,np,nξ,TS,typeof(xids),typeof(uids),
            typeof(xyids),typeof(ξids),typeof(ξxyids)}(subsystems, xids, uids,
                                                       xyids, ξids, ξxyids)
    end
end
subsystems(cs::ProductSystem) = cs.subsystems
xindex(cs::ProductSystem) = cs.xids
uindex(cs::ProductSystem) = cs.uids
xyindex(cs::ProductSystem) = cs.xyids

function dx(cs::ProductSystem{ΔT, nx, nu}, x::SVector{nx},
            u::SVector{nu}, t::AbstractFloat) where {ΔT, nx, nu}

    dx_val = MVector{nx, promote_type(eltype(x), eltype(u))}(undef)

    for (xidᵢ, uidᵢ, subᵢ) in zip(xindex(cs), uindex(cs), subsystems(cs))
        dx_val[xidᵢ] = dx(subᵢ, x[xidᵢ], u[uidᵢ], t)
    end

    return SVector{nx}(dx_val)
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
function LinearizationStyle(cs::ProductSystem)
    return (isnothing(n_linstates(cs)) ? DefaultLinearization() :
            FeedbackLinearization())
end

n_linstates(cs::ProductSystem{ΔT,nx,nu,np,nξ}) where {ΔT,nx,nu,np,nξ} = nξ
ξxyindex(cs::ProductSystem) = cs.ξxyids

# TODO: outsource some system-product function (Compose large system of subsystems)
@inline function feedback_linearized_system(cs::ProductSystem)
    nx = n_states(cs)
    nu = n_controls(cs)
    # the full matrices
    A = @MMatrix zeros(nx, nx)
    B = @MMatrix zeros(nx, nu)
    for (i, sub) in enumerate(subsystems(cs))
        xid = xindex(cs)[i]
        uid = uindex(cs)[i]

        lin_sub = feedback_linearized_system(sub)
        A[xid, xid] = lin_sub.dyn.A
        B[xid, uid] = lin_sub.dyn.B
    end
    dyn = LinearSystem{samplingtime(cs)}(SMatrix(A), SMatrix(B))
    return LTISystem(dyn, ξxyindex(cs))
end

# TODO: There might be a more compact way to do this:
# https://stackoverflow.com/questions/58616878/julia-efficiently-map-zipped-tuples-to-tuple
function x_from(cs::ProductSystem, ξ::SVector)
    subs = subsystems(cs)
    xids = xindex(cs)
    # TODO: in general: we could have nx != nξ. For now this is okay.
    ξids = xindex(cs)
    nx = n_states(cs)
    x = @MVector zeros(nx)

    for i in 1:length(subs)
        ξᵢ = ξ[ξids[i]]
        subᵢ = subs[i]
        x[xids[i]] = x_from(subᵢ, ξᵢ)
    end

    return SVector(x)
end

function ξ_from(cs::ProductSystem, x::SVector)
    subs = subsystems(cs)
    xids = xindex(cs)
    # TODO: in general: we could have nx != nξ. For now this is okay.
    ξids = xindex(cs)
    nξ = n_states(cs)
    ξ = @MVector zeros(nξ)

    for i in 1:length(subs)
        xᵢ = x[xids[i]]
        subᵢ = subs[i]
        ξ[ξids[i]] = ξ_from(subᵢ, xᵢ)
    end

    return SVector(ξ)
end

function λ_issingular(cs::ProductSystem, ξ::SVector)
    subs = subsystems(cs)
    # TODO: in general: we could have nx != nξ. For now this is okay.
    ξids = xindex(cs)
    return any(zip(subsystems(cs), ξids)) do (subᵢ, ξis)
        λ_issingular(subᵢ, ξ[ξis])
    end
end
