
# two player car dynamics
# TODO: maybe generalize to joint dynamics
struct ProductSystem{ΔT, nx, nu, xids, uids, np, TS<:NTuple{np, <:ControlSystem{ΔT}}} <: ControlSystem{ΔT, nx, nu}
    sub_systems::TS

    function ProductSystem(sub_systems::TS) where {ΔT, np, TS<:NTuple{np, <:ControlSystem{ΔT}}}
        # compute the joint state and control dimensions of the combined system
        nx, nu = 0, 0; xids, uids = [], []
        for sub in sub_systems
            nxᵢ = n_states(sub); nuᵢ = n_controls(sub)
            push!(xids, SVector{n_states(sub)}((nx+1):(nx+nxᵢ)))
            push!(uids, SVector{n_controls(sub)}((nu+1):(nu+nuᵢ)))
            nx += nxᵢ; nu += nuᵢ
        end
        new{ΔT, nx, nu, Tuple(xids), Tuple(uids), np, TS}(sub_systems)
    end
end

function dx(cs::ProductSystem{ΔT, nx, nu, xids, uids}, x::SVector{nx},
            u::SVector{nu}, t::AbstractFloat) where {ΔT, nx, nu, xids, uids}

    dx_val = MVector{nx, promote_type(eltype(x), eltype(u))}(undef)

    for (xidᵢ, uidᵢ, subᵢ) in zip(xids, uids, cs.sub_systems)
        dx_val[xidᵢ] = dx(subᵢ, x[xidᵢ], u[uidᵢ], t)
    end

    return SVector{nx}(dx_val)
end
# TODO: maybe also overload `linearize` and `integrate` to exploit sparsity
