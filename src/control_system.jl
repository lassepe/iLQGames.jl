"""
$(TYPEDEF)

The abstract type describing a control system. With state-space dimension `nx`,
control dimension `nu` and sampling time ΔT. By convention, for `ΔT == 0` the
system is continuous.

"""
abstract type ControlSystem{ΔT, nx, nu} end

"""
    $(TYPEDSIGNATURES)

Returns the sampling time ΔT of the system.
"""
samplingtime(::Type{<:ControlSystem{ΔT}}) where {ΔT} = ΔT
samplingtime(cs::ControlSystem) = samplingtime(typeof(cs))

"""
    $(TYPEDSIGNATURES)

Returns true if the system is has a non-zero sampling rate (e.g. discrete or sampled
contiuous system.
"""
issampled(cs::ControlSystem) = !iszero(samplingtime(cs))

"""
    $(TYPEDSIGNATURES)

Returns the number of states of the control system.
"""
n_states(::Type{<:ControlSystem{ΔT, nx}}) where {ΔT, nx} = nx
n_states(cs::ControlSystem) = n_states(typeof(cs))

"""
    $(TYPEDSIGNATURES)

Returns the number of controls of the control system.
"""
n_controls(::Type{<:ControlSystem{ΔT, nx, nu}}) where {ΔT, nx, nu} = nu
n_controls(cs::ControlSystem) = n_controls(typeof(cs))

"-------------------------------- Minimal Interface -------------------------------"

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

We provide a convencience default below.
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

Returns the discrete time (ZoH) Jacobian linearization for a system with
non-zero sampling time ΔT. Input arguments have the same meaning as for
`linearization`.

We provide a convencience default below.
"""
function linearize_discrete end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x0::SVector, u::SVector, t0::AbstractFloat,
                    ΔT::AbstractFloat, n_intsteps::Float64)

Integrate propagate the state `x0` given the control `u` and the current time
`t0` until `t0+ΔT`. Here, `n_intsteps` is the number of steps that is used between
`t0` and `t0+ΔT`.

We provide a convencience default below.
"""
function integrate end

"--------------------- Optional Interface Extensions ---------------------"

"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns the indeces of the x and y coordinate of this sytem as SIndex.
"""
function xyindex end

"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns the indices of the states of this system as SIndex.
"""
function xindex end

"--------------------- Convencience Implementations ---------------------"

"""
    $(TYPEDSIGNATURES)

A convencience implementation of `linearize` using `ForwardDiff.jl`. Overload
this with an explicit version to get better performance.
"""
function _linearize_ad(cs::ControlSystem, x::SVector, u::SVector, t::AbstractFloat)
    @warn "You are using the fallback linearization using ForwardDiff. Consider
    implementing a custom `linearize` or `linearize_discrete` for your
    `ControlSystem` type." maxlog=1
    A = ForwardDiff.jacobian(x->dx(cs, x, u, t), x)
    B = ForwardDiff.jacobian(u->dx(cs, x, u, t), u)
    return LinearSystem{0}(A, B)
end
function linearize(cs::ControlSystem, x::SVector, u::SVector, t::AbstractFloat)
    return _linearize_ad(cs, x, u, t)
end

"""
    $(TYPEDSIGNATURES)

A convencience implementation of `linearize_discrete`. For better performance,
this may be overloaded with some explicit (analytic) expressions (that may even
avoid calling `linearize`).
"""
linearize_discrete(cs::ControlSystem,
                   x0::SVector,
                   u0::SVector,
                   t0::AbstractFloat) = discretize(linearize(cs, x0, u0, t0),
                                                   Val{samplingtime(cs)}())

"""
    $(TYPEDSIGNATURES)

A convencience implementation of `integrate` using RungeKutta of order 4 as
convencience default. You may impelement your own integrator here. Consider
using `DifferentialEquations.jl.
"""
function integrate(cs::ControlSystem, x0::SVector, u::SVector, t0::AbstractFloat,
                   ΔT::AbstractFloat, n_intsteps::Int=2)
    Δt = ΔT/n_intsteps
    x = x0
    for t in range(t0, stop=t0+ΔT, length=n_intsteps+1)[1:end-1]
        k1 = Δt * dx(cs, x, u, t);
        k2 = Δt * dx(cs, x + 0.5 * k1, u, t + 0.5 * Δt);
        k3 = Δt * dx(cs, x + 0.5 * k2, u, t + 0.5 * Δt);
        k4 = Δt * dx(cs, x + k3      , u, t + Δt);
        x += (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
    end
    return x
end

"""
    $(TYPEDSIGNATURES)

Integrate to xₖ₊₁ starting from x, applying u.
"""
function next_x(cs::ControlSystem, x::SVector, u::SVector, t::Float64)
    @assert issampled(cs) "next_x requires `ControlSystem` with ΔT > 0."
    return integrate(cs, x, u, t, samplingtime(cs))
end
