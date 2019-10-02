# Maybe this should know about t0 as well.
struct SystemTrajectory{h, ΔT, nx, nu, TX<:SizedVector{h,<:SVector{nx}},
                        TU<:SizedVector{h,<:SVector{nu}}}
    "The sequence of states."
    x::TX
    "The sequence of controls."
    u::TU
    # TODO: also add t0 (initial time)
end
function SystemTrajectory{ΔT}(x::TX, u::TU) where {h, ΔT, nx, nu,
                                                   TX<:SizedVector{h,<:SVector{nx}},
                                                   TU<:SizedVector{h,<:SVector{nu}}}
    return SystemTrajectory{h, ΔT, nx, nu, TX, TU}(x, u)
end
sampling_time(::SystemTrajectory{h, ΔT}) where {h, ΔT} = ΔT
horizon(::SystemTrajectory{h}) where {h} = h
time_disc2cont(traj::SystemTrajectory, k::Int) = (k-1)*sampling_time(traj)
timepoints(traj::SystemTrajectory) = (time_disc2cont(traj, k) for k in
                                      1:horizon(traj))

# thin interface with Base for convenience
function Base.zero(::Type{<:SystemTrajectory{h, ΔT, nx, nu}}) where{h, ΔT, nx, nu}
    return SystemTrajectory{ΔT}(zero(SizedVector{h, SVector{nx, Float64}}),
                                zero(SizedVector{h, SVector{nu, Float64}}))
end
Base.zero(t::SystemTrajectory) = zero(typeof(t))
Base.copy(t::SystemTrajectory) = SystemTrajectory{sampling_time(t)}(copy(t.x),
                                                                    copy(t.u))
