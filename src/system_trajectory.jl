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
Base.zero(::Type{<:SystemTrajectory{h, ΔT, nx, nu}}) where{h, ΔT, nx, nu} = SystemTrajectory{ΔT}(zero(SizedVector{h, SVector{nx, Float64}}),
                         zero(SizedVector{h, SVector{nu, Float64}}))
