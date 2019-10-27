"Computes the joint state and control dimensions of the combined system."
function xu_dims(sub_systems)
    nx, nu = 0, 0; xids, uids = [], []
    for sub in sub_systems
        nxᵢ = n_states(sub); nuᵢ = n_controls(sub)
        push!(xids, SVector{n_states(sub)}((nx+1):(nx+nxᵢ)))
        push!(uids, SVector{n_controls(sub)}((nu+1):(nu+nuᵢ)))
        nx += nxᵢ; nu += nuᵢ
    end

    return nx, nu, xids, uids
end

struct ProductSystem{ΔT,nx,nu,xids,uids,xyids,np,
                     TS<:NTuple{np,<:ControlSystem{ΔT}}}<:ControlSystem{ΔT, nx, nu}
    sub_systems::TS

    function ProductSystem(sub_systems::TS) where {ΔT, np, TS<:NTuple{np, <:ControlSystem{ΔT}}}
        nx, nu, xids, uids = xu_dims(sub_systems)
        xyids = Tuple(xid[xyindex(sub)] for (xid, sub) in zip(xids, sub_systems))
        new{ΔT,nx,nu,Tuple(xids),Tuple(uids),xyids,np,TS}(sub_systems)
    end
end
xindex(cs::ProductSystem{ΔT,nx,nu,xids}) where {ΔT,nx,nu,xids} = xids
uindex(cs::ProductSystem{ΔT,nx,nu,xids,uids}) where {ΔT,nx,nu,xids,uids} = uids
xyindex(cs::ProductSystem{ΔT,nx,nu,xids,uids,xyids}) where {ΔT,nx,nu,xids,uids,xyids} = xyids
sub_systems(cs::ProductSystem) = cs.sub_systems

function dx(cs::ProductSystem{ΔT, nx, nu, xids, uids}, x::SVector{nx},
            u::SVector{nu}, t::AbstractFloat) where {ΔT, nx, nu, xids, uids}

    dx_val = MVector{nx, promote_type(eltype(x), eltype(u))}(undef)

    for (xidᵢ, uidᵢ, subᵢ) in zip(xids, uids, cs.sub_systems)
        dx_val[xidᵢ] = dx(subᵢ, x[xidᵢ], u[uidᵢ], t)
    end

    return SVector{nx}(dx_val)
end

# computing large matrix exponentials is expensive. Therefore, we exploit the
# sparsity of the dynamics and compute the linearization for each sub_systems
@inline function linearize_discrete(cs::ProductSystem, x::SVector, u::SVector, t::AbstractFloat)
    nx = n_states(cs)
    nu = n_controls(cs)
    # the full matrices
    A = @MMatrix zeros(nx, nx)
    B = @MMatrix zeros(nx, nu)
    for (i, sub) in enumerate(sub_systems(cs))
        xid = xindex(cs)[i]
        uid = uindex(cs)[i]

        lin_sub = linearize_discrete(sub, x[xid], u[uid], t)
        A[xid, xid] = lin_sub.A
        B[xid, uid] = lin_sub.B
    end
    return LinearSystem{samplingtime(cs)}(SMatrix(A), SMatrix(B))
end
