using DocStringExtensions
using StaticArrays

"""
$(TYPEDEF)

A lorenz attractor with with two inputs and parameters `σ`, `ρ` and `β`.
"""
struct Lorenz3D{T1, T2, T3} <: ControlSystem{3, 2}
    σ::T1;
    ρ::T2;
    β::T3;
end

function dx(cs::Lorenz3D, x::SVector{3}, u::SVector{2}, t::Real)
    dx1 = cs.σ*(x[2] - x[1])
    dx2 = x[1]*(cs.ρ - x[3]) - x[2] + u[2]
    dx3 = x[1]*x[2] - cs.β*x[3] + u[1]

    return @SVector[dx1, dx2, dx3]
end

"""
$(TYPEDEF)

A simple dubins car model.

# Fields

$(TYPEDFIELDS)
"""
struct Car5D <: ControlSystem{5, 2}
    "inter-axle length (m)"
    l::Float64
end

function dx(cs::Car5D, x::SVector{5}, u::SVector{2}, t::Real)
    # position: x
    dx1 = x[5] * cos(x[3])
    # position: y
    dx2 = x[5] * sin(x[3])
    # orientation
    dx3 = x[5] * tan(x[4]) / cs.l
    # steering angle
    dx4 = u[1]
    # speed
    dx5 = u[2]

    return @SVector[dx1, dx2, dx3, dx4, dx5]
end
