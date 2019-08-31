"""
$(TYPEDEF)

The abstract type describing a control system. With state-space dimension `nx`
and control dimension `nu`.
"""
# TODO: maybe we don't even need the type parametrization
abstract type ControlSystem{nx, nu} end

"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns the number of states of the control system.
"""
n_states(::Type{<:ControlSystem{nx, nu}}) where {nx, nu} = nx
n_states(cs::ControlSystem) = n_states(typeof(cs))

"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns the number of controls of the control system.
"""
n_controls(::Type{<:ControlSystem{nx, nu}}) where {nx, nu} = nu
n_controls(cs::ControlSystem) = n_states(typeof(cs))

""" $(FUNCTIONNAME)(cs::ControlSystem, x::SVector, u::SVector, t::AbstractFloat)
Returns the time derivative of the state `dx` at a given state `x`, control
input `u` and time `t`
"""
function dx end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x::SVector, u::SVector, t::AbstractFloat)

Returns the contiuous time Jacobian linearization of the dynamics defined in
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

struct SystemTrajectory{h, TX<:SizedVector{h,<:SVector},
                        TU<:SizedVector{h,<:SVector}, TF<:AbstractFloat}
    "The sequence of states."
    x::TX
    "The sequence of controls."
    u::TU
    "The time step."
    ΔT::TF
end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x::SVector, u::SVector, t::AbstractFloat)

A convencience implementaiton of `linearize` using `ForwardDiff.jl`. Overload
this with an explicit version to get better performance.
"""
function linearize(cs::ControlSystem, x::SVector, u::SVector, t::AbstractFloat)
    A = ForwardDiff.jacobian(x->dx(cs, x, u, t), x)
    B = ForwardDiff.jacobian(u->dx(cs, x, u, t), u)
    return LinearSystem(A, B)
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
