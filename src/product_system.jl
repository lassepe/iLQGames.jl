"Computes the joint state and control dimensions of the combined system."
function xu_dims(subsystems)
    nx, nu = 0, 0; xids, uids = [], []
    for sub in subsystems
        nxᵢ = n_states(sub); nuᵢ = n_controls(sub)
        push!(xids, SVector{n_states(sub)}((nx+1):(nx+nxᵢ)))
        push!(uids, SVector{n_controls(sub)}((nu+1):(nu+nuᵢ)))
        nx += nxᵢ; nu += nuᵢ
    end

    return nx, nu, Tuple(xids), Tuple(uids)
end

struct ProductSystem{ΔT,nx,nu,np,TS<:NTuple{np,<:ControlSystem{ΔT}},TXI<:NTuple{np},
                     TUI<:NTuple{np},TXY<:NTuple{np}}<:ControlSystem{ΔT, nx, nu}
    subsystems::TS
    xids::TXI
    uids::TUI
    xyids::TXY
    function ProductSystem(subsystems::TS) where {ΔT,np,TS<:NTuple{np,<:ControlSystem{ΔT}}}
        nx, nu, xids, uids = xu_dims(subsystems)
        xyids = Tuple(xid[xyindex(sub)] for (xid, sub) in zip(xids, subsystems))
        new{ΔT,nx,nu,np,TS,typeof(xids),typeof(uids),typeof(xyids)}(subsystems, xids, uids, xyids)
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
Base.@pure function LinearizationStyle(cs::ProductSystem)
    linstyles = unique(LinearizationStyle.(cs.subsystems))
    return length(linstyles) == 1 ? linstyles[1] : return DefaultLinearization()
end


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
        A[xid, xid] = lin_sub.A
        B[xid, uid] = lin_sub.B
    end
    return LinearSystem{samplingtime(cs)}(SMatrix(A), SMatrix(B))
end
