"""
$(TYPEDEF)

A lorenz attractor with with two inputs and parameters `σ`, `ρ` and `β`.
"""
struct Lorenz3D{ΔT} <: ControlSystem{ΔT, 3, 2}
    σ::Float64;
    ρ::Float64;
    β::Float64;
end

function dx(cs::Lorenz3D, x::SVector{3}, u::SVector{2}, t::AbstractFloat)
    dx1 = cs.σ*(x[2] - x[1])
    dx2 = x[1]*(cs.ρ - x[3]) - x[2] + u[2]
    dx3 = x[1]*x[2] - cs.β*x[3] + u[1]

    return @SVector[dx1, dx2, dx3]
end

