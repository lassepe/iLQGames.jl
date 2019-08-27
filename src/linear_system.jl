"""
$(TYPEDEF)

Represents a simple control system with linear dynamics.

# Parameters:

- `nx`: the number of states
- `nu`: the number of inputs

# Fields

$(TYPEDFIELDS)
"""
struct LinearSystem{nx, nu, TA<:SMatrix{nx, nx}, TB<:SMatrix{nx, nu}} <: ControlSystem{nx, nu}
    "The state transitioni matrix"
    A::TA
    "The control input matrix"
    B::TB
end
