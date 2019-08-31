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

dx(ls::LinearSystem, x::SVector, u::SVector, t::AbstractFloat) = ls.A*x + ls.B*u
linearize(ls::LinearSystem, x::SVector, u::SVector, t::AbstractFloat) = ls

"""
    $(FUNCTIONNAME)(ls::LinearSystem, ΔT::AbstractFloat)

Computes the zero-order-hold discretization of the linear system ls with time
discretization step ΔT.
"""
function discretize(ls::LinearSystem, ΔT::AbstractFloat)
    # the discrete time system matrix
    Φ = exp(ls.A*ΔT)
    # the discrete time input matrix
    # TODO what happens if A is singular?
    Γ = inv(ls.A) * (Φ - I) * ls.B

    # TODO maybe this should be a different type of a template parameter that
    # indicates, that this is now discrete.
    return LinearSystem(Φ, Γ)
end

function discretize_exp(ls::LinearSystem{nx, nu}, ΔT::Float64) where {nx, nu}
    M = vcat([ls.A ls.B], @SMatrix(zeros(nu, nu+nx)))
    #M = vcat([A B], SMatrix{nu, nx+nu, Float64, nu*(nx+nu)}(zeros(nu, nx+nu)))

    eMT = exp(M*ΔT)
    rx = SVector{nx}(1:nx)
    ru = SVector{nu}((nx+1):(nx+nu))

    Φ = eMT[rx, rx]
    Γ = eMT[rx, ru]

    return LinearSystem(Φ, Γ)
end
