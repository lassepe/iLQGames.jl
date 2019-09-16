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

"""
$(TYPEDEF)

A simple dubins car model.

# Fields

$(TYPEDFIELDS)
"""
struct Car5D{ΔT} <: ControlSystem{ΔT, 5, 2}
    "inter-axle length (m)"
    l::Float64
end

function dx(cs::Car5D, x::SVector{5}, u::SVector{2}, t::AbstractFloat)
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

function linearize_discrete(cs::Car5D, x::SVector{5}, u::SVector{2}, t::AbstractFloat)
    ΔT = sampling_time(cs)
    cθ = cos(x[3]) * ΔT
    sθ = sin(x[3]) * ΔT
    cϕ = cos(x[4])
    tϕ = tan(x[4])

    A = @SMatrix [1 0 -x[5]*sθ 0                    cθ;
                  0 1 x[5]*cθ  0                    sθ;
                  0 0 1        x[5]*ΔT/(cs.l*cϕ*cϕ) tϕ*ΔT/cs.l;
                  0 0 0        1                    0;
                  0 0 0        0                    1]

    B = @SMatrix [0  0;
                  0  0;
                  0  0;
                  ΔT 0;
                  0  ΔT];
    return LinearSystem{ΔT}(A, B)
end
