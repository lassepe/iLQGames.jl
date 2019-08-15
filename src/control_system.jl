"""
$(TYPEDEF)

The abstract type describing a control system. With state-space dimension `NX`
and control dimension `NU`.
"""
# TODO: maybe we don't even need the type parametrization
abstract type ControlSystem{NX, NU} end

"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns the number of states of the control system.
"""
dim_x(cs::ControlSystem{NX, NU}) where {NX, NU} = NX

"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns the number of controls of the control system.
"""
dim_u(cs::ControlSystem{NX, NU}) where {NX, NU} = NU

""" $(FUNCTIONNAME)(cs::ControlSystem, x::AbstractVector, u::AbstractVector, t::Real)
Returns the time derivative of the state `dx` at a given state `x`, control
input `u` and time `t`
"""
function dx end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x::AbstractVector, u::AbstractVector, t::Real)

Returns the linearization of the dynamics defined in `dx` at a given state `x`,
control `u` and time point `t` in terms of the A and B matrix.
(such that ẋ = A*x + B*u)

# Convenience Default:

We provide a default using `ForwardDiff.jl`. Overload this with an explicit
version to get better performance.
"""
function linearize(cs::ControlSystem, x::AbstractVector, u::AbstractVector, t::Real)
    A = ForwardDiff.jacobian(x->dx(cs, x, u, t), x)
    B = ForwardDiff.jacobian(u->dx(cs, x, u, t), u)
    # TODO maybe this should return an LTI system
    return A, B
end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x0::AbstractVector, u::AbstractVector, t0::Real, ΔT::Real)

Integrate propagate the state `x0` given the control `u` and the current time
`t0` until `t0+ΔT`. Here, `k` is the number of steps that is used between `t0` and
`t0+ΔT`.

# Convenience Default:

Here we provide RungeKutta of order 4 as convencience default. You may
impelement your own integrator here. Consider using `DifferentialEquations.jl.
"""
function integrate(cs::ControlSystem, x0::AbstractVector, u::AbstractVector, t0::Real, ΔT::Real, k::Int=2)
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

# TODO: Maybe we don't need a linear system type. Seems redundant here.
struct LTISystem{NX, NU, TA<:AbstractArray, TB<:AbstractArray} <: ControlSystem{NX, NU}
    A::TA
    B::TB
end
dx(cs::LTISystem, x, u, t=0) = A * x + B * u
linearize(cs::LTISystem{nx, nu, TA, TB}, x=zeros(nx), u=zeros(nu), t=0) where {nx, nu, TA, TB} = A, B
