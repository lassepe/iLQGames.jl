"""
$(TYPEDEF)

Represents a simple control system with linear dynamics.

# Parameters:

- `ΔT`: the sampling time of the system (se)
- `nx`: the number of states
- `nu`: the number of inputs

# Fields

$(TYPEDFIELDS)
"""
struct LinearSystem{ΔT, nx, nu, TA<:SMatrix{nx, nx}, TB<:SMatrix{nx, nu}} <: ControlSystem{ΔT, nx, nu}
    "The state transition matrix"
    A::TA
    "The control input matrix"
    B::TB
end

LinearSystem{ΔT}(A::TA, B::TB) where {ΔT, nx, nu, TA<:SMatrix{nx, nx}, TB<:SMatrix{nx, nu}} = LinearSystem{ΔT, nx, nu, TA, TB}(A, B)

dx(ls::LinearSystem, x::SVector, u::SVector, t::AbstractFloat)  = begin @assert !issampled(ls); ls.A*x + ls.B*u end
next_x(ls::LinearSystem, x::SVector, u::SVector) = begin @assert issampled(ls); ls.A*x + ls.B*u end
linearize(ls::LinearSystem, x::SVector, u::SVector, t::AbstractFloat) = ls

"""
    $(FUNCTIONNAME)(ls::LinearSystem, ΔT::AbstractFloat)

Computes the zero-order-hold discretization of the linear system ls with time
discretization step ΔT.
"""
function discretize(ls::LinearSystem, ΔT::AbstractFloat)
    @assert !issampled(ls) "Can't discretize a discrete system."

    # the discrete time system matrix
    Φ = exp(ls.A*ΔT)
    # the discrete time input matrix
    # TODO what to do if A is singular?
    Γ = inv(ls.A) * (Φ - I) * ls.B

    return LinearSystem{ΔT}(Φ, Γ)
end

#function discretize_exp(ls::LinearSystem{nx, nu}, ΔT::Float64) where {nx, nu}
#    @assert !issampled(ls) "Can't discretize a discrete system."
#
#    M = vcat([ls.A ls.B], @SMatrix(zeros(nu, nu+nx)))
#    #M = vcat([A B], SMatrix{nu, nx+nu, Float64, nu*(nx+nu)}(zeros(nu, nx+nu)))
#
#    eMT = exp(M*ΔT)
#    rx = SVector{nx}(1:nx)
#    ru = SVector{nu}((nx+1):(nx+nu))
#
#    Φ = eMT[rx, rx]
#    Γ = eMT[rx, ru]
#
#    return LinearSystem{ΔT}(Φ, Γ)
#end

struct LTVSystem{h, ΔT, nx, nu, TD<:SizedVector{h, <:LinearSystem{ΔT, nx, nu}}} <: ControlSystem{ΔT, nx, nu} "The discrete time series of linear systems."
    dyn::TD

    LTVSystem(dyn::TD) where {h, ΔT, nx, nu, TD<:SizedVector{h, <:LinearSystem{ΔT, nx, nu}}} = begin
        @assert ΔT > 0 "LTVSystem require finite discretization steps."
        new{h, ΔT, nx, nu, TD}(dyn)
    end
end
Base.eltype(::Type{<:LTVSystem{h, ΔT, nx, nu, TD}}) where {h, ΔT, nx, nu, TD} = eltype(TD)
Base.getindex(ds::LTVSystem, i) = getindex(ds.dyn, i)
next_x(cs::LTVSystem, xₖ::SVector, uₖ::SVector, k::Int) = next_x(cs.dyn[k], xₖ, uₖ)

struct SystemTrajectory{h, ΔT, nx, nu, TX<:SizedVector{h,<:SVector{nx}},
                        TU<:SizedVector{h,<:SVector{nu}}}
    "The sequence of states."
    x::TX
    "The sequence of controls."
    u::TU
end
SystemTrajectory{ΔT}(x::TX, u::TU) where {h, ΔT, nx, nu,
                                          TX<:SizedVector{h,<:SVector{nx}},
                                          TU<:SizedVector{h,<:SVector{nu}}} = SystemTrajectory{h, ΔT, nx, nu, TX, TU}(x, u)
sampling_time(t::SystemTrajectory{h, ΔT}) where {h, ΔT} = ΔT
Base.zero(::Type{<:SystemTrajectory{h, ΔT, nx, nu}}) where{h, ΔT, nx, nu} = SystemTrajectory{ΔT}(zero(SizedVector{h, SVector{nx, Float64}}),
                         zero(SizedVector{h, SVector{nu, Float64}}))


function trajectory!(traj::SystemTrajectory{h}, cs::ControlSystem,
                     γ::SizedVector{h, <:AffineStrategy},
                     last_op::SystemTrajectory{h}, x0::SVector) where {h}

    @assert sampling_time(traj) == sampling_time(last_op) == sampling_time(cs)

    # TODO: think about in which cases this can be first(last_op.x)
    xₖ = x0
    # xₖ = first(last_op.x)

    for k in 1:h
        # the quantities on the old operating point
        x̃ₖ = last_op.x[k]
        ũₖ = last_op.u[k]
        # the current strategy
        γₖ = γ[k]
        # the deviation from the last operating point
        Δxₖ = xₖ - x̃ₖ

        # record the new operating point:
        x_opₖ = traj.x[k] = xₖ
        u_opₖ = traj.u[k] = control_input(γₖ, Δxₖ, ũₖ)

        # integrate x forward in time for the next iteration.
        xₖ = next_x(cs, x_opₖ, u_opₖ, k)
    end
    return traj
end
