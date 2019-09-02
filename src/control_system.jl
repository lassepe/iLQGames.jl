"""
$(TYPEDEF)

The abstract type describing a control system. With state-space dimension `nx`,
control dimension `nu` and sampling time ΔT. By convention, for `ΔT == 0` the
system is continuous.

"""
# TODO: maybe we don't even need the type parametrization
abstract type ControlSystem{ΔT, nx, nu} end

"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns the sampling time ΔT of the system.
"""
sampling_time(cs::ControlSystem{ΔT}) where {ΔT} = ΔT

"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns true if the system is has a non-zero sampling rate (e.g. discrete or sampled contiuous system.
"""
issampled(cs::ControlSystem) = !iszero(sampling_time(cs))

"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns the number of states of the control system.
"""
n_states(::Type{<:ControlSystem{ΔT, nx}}) where {ΔT, nx} = nx
n_states(cs::ControlSystem) = n_states(typeof(cs))

"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns the number of controls of the control system.
"""
n_controls(::Type{<:ControlSystem{ΔT, nx, nu}}) where {ΔT, nx, nu} = nu
n_controls(cs::ControlSystem) = n_states(typeof(cs))

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x::SVector, u::SVector, t::AbstractFloat)

Returns the time derivative of the state `dx` at a given state `x`, control
input `u` and time `t`
"""
function dx end

""""
    $(FUNCTIONNAME)(cs::ControlSystem, xₖ::SVector, uₖ::SVector, k::Int)

Returns the next state (`xₖ₊₁`) for a control system with non-zero `ΔT` when
applying input `uₖ` at state `xₖ` and time step k
"""
function next_x end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x::SVector, u::SVector, t::AbstractFloat)

Returns the continuous time Jacobian linearization of the dynamics defined in
`dx` at a given state `x`, control `u` and time point `t` in terms of a
`LinearSystem` (A, B).  (such that Δẋ ≈ A*Δx + B*Δu)

We provide a convencience default below.
"""
function linearize end

""""
    $(FUNCTIONNAME)(cs::ControlSystem, x::SVector, u::SVector, t::AbstractFloat)

Returns the discrete time (ZoH) Jacobian linearization. Input arguments
have the same meaning as for `linearization`; with additional input
argument `ΔT`, the time step for discretization.

We provide a convencience default below.
"""
function linearize_discrete end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x0::SVector, u::SVector, t0::AbstractFloat, ΔT::AbstractFloat)

Integrate propagate the state `x0` given the control `u` and the current time
`t0` until `t0+ΔT`. Here, `k` is the number of steps that is used between `t0` and
`t0+ΔT`.

We provide a convencience default below.
"""
function integrate end


"--------------------- Convencience Impelemtnations ---------------------"

struct DiscreteTimeVaryingSystem{h, ΔT, nx, nu, TD<:SizedVector{h, <:ControlSystem{ΔT, nx, nu}}} <: ControlSystem{ΔT, nx, nu}
    "The discrete time series of linear systems."
    dyn::TD

    DiscreteTimeVaryingSystem(dyn::TD) where {h, ΔT, nx, nu, TD<:SizedVector{h, <:ControlSystem{ΔT, nx, nu}}} = begin
        @assert ΔT > 0 "DiscreteTimeVaryingSystem require finite discretization steps."
        new{h, ΔT, nx, nu, TD}(dyn)
    end
end
Base.getindex(ds::DiscreteTimeVaryingSystem, i) = getindex(ds.dyn, i)
next_x(cs::DiscreteTimeVaryingSystem, xₖ::SVector, uₖ::SVector, k::Int) = next_x(cs.dyn[k], xₖ, uₖ, (k-1)*sampling_time(cs))

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

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x::SVector, u::SVector, t::AbstractFloat)

A convencience implementaiton of `linearize` using `ForwardDiff.jl`. Overload
this with an explicit version to get better performance.
"""
function linearize(cs::ControlSystem, x::SVector, u::SVector, t::AbstractFloat)
    A = ForwardDiff.jacobian(x->dx(cs, x, u, t), x)
    B = ForwardDiff.jacobian(u->dx(cs, x, u, t), u)
    return LinearSystem{0}(A, B)
end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x0::SVector, u0::SVector, t0::AbstractFloat, ΔT::AbstractFloat)

A convencience implementaiton of `linearize_discrete`. For better performance,
this may be overloaded with some explicit (analytic) expressions (that may even
avoid calling `linearize`).
"""
linearize_discrete(cs::ControlSystem,
                   x0::SVector,
                   u0::SVector,
                   t0::AbstractFloat, ΔT::AbstractFloat) = discretize(linearize(cs, x0, u0, t0), ΔT)

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x0::SVector, u::SVector, t0::AbstractFloat, ΔT::AbstractFloat)

A convencience implementaiton of `integrate` using RungeKutta of order 4 as
convencience default. You may impelement your own integrator here. Consider
using `DifferentialEquations.jl.
"""
function integrate(cs::ControlSystem, x0::SVector, u::SVector, t0::AbstractFloat, ΔT::AbstractFloat, k::Int=2)
    @assert iszero(t0) "currently there are parts of the code that don't handle
                       t0!=0 correctly so this shoudl not be used"
    Δt = ΔT/k
    x = x0
    for t in range(t0, stop=t0+ΔT, length=k+1)[1:end-1]
        k1 = Δt * dx(cs, x, u, t);
        k2 = Δt * dx(cs, x + 0.5 * k1, u, t + 0.5 * Δt);
        k3 = Δt * dx(cs, x + 0.5 * k2, u, t + 0.5 * Δt);
        k4 = Δt * dx(cs, x + k3      , u, t + Δt);
        x += (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
    end
    return x
end

function trajectory!(traj::SystemTrajectory{h}, cs::DiscreteTimeVaryingSystem{h},
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
