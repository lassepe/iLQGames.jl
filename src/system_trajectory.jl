# Maybe this should know about t0 as well.
struct SystemTrajectory{h, ΔT, nx, nu, TX<:SizedVector{h,<:SVector{nx}},
                        TU<:SizedVector{h,<:SVector{nu}}}
    "The sequence of states."
    x::TX
    "The sequence of controls."
    u::TU
    # TODO: also add t0 (initial time)
end
SystemTrajectory{ΔT}(x::TX, u::TU) where {h, ΔT, nx, nu,
                                          TX<:SizedVector{h,<:SVector{nx}},
                                          TU<:SizedVector{h,<:SVector{nu}}} = SystemTrajectory{h, ΔT, nx, nu, TX, TU}(x, u)
sampling_time(t::SystemTrajectory{h, ΔT}) where {h, ΔT} = ΔT
horizon(t::SystemTrajectory{h}) where {h} = h
contiuous_time(t::SystemTrajectory, k::Int) = (k-1)*sampling_time(t)
timepoints(t::SystemTrajectory) = (contiuous_time(t, k) for k in 1:horizon(t))

Base.zero(::Type{<:SystemTrajectory{h, ΔT, nx, nu}}) where{h, ΔT, nx, nu} = SystemTrajectory{ΔT}(zero(SizedVector{h, SVector{nx, Float64}}),
                         zero(SizedVector{h, SVector{nu, Float64}}))
Base.zero(t::SystemTrajectory) = zero(typeof(t))
Base.copy(t::SystemTrajectory) = SystemTrajectory{sampling_time(t)}(copy(t.x), copy(t.u))
