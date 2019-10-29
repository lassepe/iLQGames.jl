""""--------------------- The Feedback Linearization Interface ---------------------

 NOTE: Here we don't implement the interface for cost transformation (e.g. ∂l/∂ξ).
 Instead, the problem write should provide the transfomed cost directly.
"""


"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns a discretized LinearSystem that descibes the dynamics of the linearized
state coordinate.

NOTE: this must not be used with the non-linear state coordinates.
"""
#TODO: maybe wrap state type for safety.
function feedback_linearized_system end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, ξ)

The state conversion map `λ(ξ)`. Transforms linear states `ξ` to nonlinear states
`x`.
"""

function x_from end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x)

Inverse state conversion map `λ⁻¹(x)`. Transforms nonlinear states `x` to linear
states `ξ`.
"""
function ξ_from end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, ξ)

Returns true if the state converstion map is singular at the given *linear* system
states `ξ`.
"""
function λ_issingular end
