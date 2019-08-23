"""
    $(FUNCTIONNAME)(cs::ControlSystem, x::AbstractVector, u::AbstractVector, t::Real)

A convencience implementaiton of `linearize` using `ForwardDiff.jl`. Overload
this with an explicit version to get better performance.
"""
function linearize(cs::ControlSystem, x::AbstractVector, u::AbstractVector, t::Real)
    A = ForwardDiff.jacobian(x->dx(cs, x, u, t), x)
    B = ForwardDiff.jacobian(u->dx(cs, x, u, t), u)
    return LinearSystem(A, B)
end

"""
    $(FUNCTIONNAME)(cs::ControlSystem, x0::AbstractVector, u::AbstractVector, t0::Real, ΔT::Real)

A convencience implementaiton of `integrate` using RungeKutta of order 4 as
convencience default. You may impelement your own integrator here. Consider
using `DifferentialEquations.jl.
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
