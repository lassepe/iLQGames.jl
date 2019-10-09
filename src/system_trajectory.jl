# Maybe this should know about t0 as well.
struct SystemTrajectory{h, ΔT, nx, nu, TX<:SizedVector{h,<:SVector{nx}},
                        TU<:SizedVector{h,<:SVector{nu}}}
    "The sequence of states."
    x::TX
    "The sequence of controls."
    u::TU
    "The start time of this trajectory."
    t0::Float64
end
function SystemTrajectory{ΔT}(x::TX, u::TU, t0::Float64) where {h, ΔT, nx, nu,
                                                                TX<:SizedVector{h,<:SVector{nx}},
                                                                TU<:SizedVector{h,<:SVector{nu}}}
    return SystemTrajectory{h, ΔT, nx, nu, TX, TU}(x, u, t0)
end
samplingtime(::SystemTrajectory{h, ΔT}) where {h, ΔT} = ΔT
horizon(::SystemTrajectory{h}) where {h} = h
initialtime(traj::SystemTrajectory) = traj.t0
function time_disc2cont(traj::SystemTrajectory, k::Int)
    @assert 1 <= k <= horizon(traj)
    return initialtime(traj) + (k-1)*samplingtime(traj)
end

function time_cont2disc(traj::SystemTrajectory, t::Float64)
    @assert t >= initialtime(traj)
    k = 1 + round(Int, (t-initialtime(traj))/samplingtime(traj))
    @assert k <= horizon(traj)
    return k
end

timepoints(traj::SystemTrajectory) = (time_disc2cont(traj, k) for k in
                                      1:horizon(traj))

# thin interface with Base for convenience
function Base.zero(::Type{<:SystemTrajectory{h, ΔT, nx, nu}}, t0::Float64=0.) where{h, ΔT, nx, nu}
    return SystemTrajectory{ΔT}(zero(SizedVector{h, SVector{nx, Float64}}),
                                zero(SizedVector{h, SVector{nu, Float64}}), t0)
end
Base.zero(traj::SystemTrajectory) = zero(typeof(traj), initialtime(traj))
function Base.copy(traj::SystemTrajectory)
    return SystemTrajectory{samplingtime(traj)}(copy(traj.x), copy(traj.u),
                                                initialtime(traj))
end
