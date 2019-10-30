""""

 NOTE: Here we don't implement the interface for cost transformation (e.g. ∂l/∂ξ).
 Instead, the problem write should provide the transfomed cost directly.

 Feedback linearizable systems shall implement this interface and join the trait.
"""

# Holy traits for linearization style dispatch
abstract type LinearizationStyle end
# systems without a special structure (e.g. use jacobian linearization)
struct JacobianLinearization <: LinearizationStyle end
# systems that are feedback linearizable (i.e. differentially flat systems)
struct FeedbackLinearization <: LinearizationStyle end
# systems that are trivially linearized (e.g. already linear)
struct TrivialLinearization <: LinearizationStyle end

LinearizationStyle(cs) = JacobianLinearization()

"--------------------- The Feedback Linearization Interface ---------------------"

"""
    $(FUNCTIONNAME)(cs::ControlSystem)

Returns the number of ξ states (linearized states).
"""
function n_linstates end

"""
$(FUNCTIONNAME)(cs::ControlSystem)

Returns the indices of the cartesian x and y coordinates in the linearized state
vector ξ.
"""
# TODO, technically this could be computed by propagating xyindex(cs) through
# ξ_from.
function ξxyindex end

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
