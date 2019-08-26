using DocStringExtensions

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

""" $(FUNCTIONNAME)(cs::ControlSystem, x::SVector, u::SVector, t::Real)
Returns the time derivative of the state `dx` at a given state `x`, control
input `u` and time `t`
"""
function dx end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x::SVector, u::SVector, t::Real)

Returns the linearization of the dynamics defined in `dx` at a given state `x`,
control `u` and time point `t` in terms of a `LinearSystem`.
(such that ẋ = A*x + B*u)

We provide a convencience default in `control_system_convenience.jl`.
"""
function linearize end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x0::SVector, u::SVector, t0::Real, ΔT::Real)

Integrate propagate the state `x0` given the control `u` and the current time
`t0` until `t0+ΔT`. Here, `k` is the number of steps that is used between `t0` and
`t0+ΔT`.

We provide a convencience default in `control_system_convenience.jl`.
"""
function integrate end
