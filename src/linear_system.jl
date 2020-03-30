"""
$(TYPEDEF)

Represents a simple control system with linear dynamics.

# Parameters:

- `ΔT`: the sampling time of the system (se)
- `nx`: the number of states
- `nu`: the number of inputs

# Fields

$(TYPEDFIELDS)
"""
struct LinearSystem{ΔT, nx, nu, TA<:SMatrix{nx, nx}, TB<:SMatrix{nx, nu}} <: ControlSystem{ΔT, nx, nu}
    "The state transition matrix"
    A::TA
    "The control input matrix"
    B::TB
end

LinearSystem{ΔT}(A::TA, B::TB) where {ΔT, nx, nu, TA<:SMatrix{nx, nx}, TB<:SMatrix{nx, nu}} = LinearSystem{ΔT, nx, nu, TA, TB}(A, B)
LinearizationStyle(::LinearSystem) = TrivialLinearization()

dx(ls::LinearSystem, x::SVector, u::SVector, t::AbstractFloat)  = begin @assert !issampled(ls); ls.A*x + ls.B*u end
next_x(ls::LinearSystem, x::SVector, u::SVector) = begin @assert issampled(ls); ls.A*x + ls.B*u end
linearize(ls::LinearSystem, x::SVector, u::SVector, t::AbstractFloat) = ls
function discretize(ls::LinearSystem, vt::Val{ΔT}) where {ΔT}
    @assert !issampled(ls) "Can't discretize a discrete system."
    @assert ΔT > 0 "Discrtization requires finite sampling time ΔT."
    # use the euler approximation to avoid expensive matrix exponential
    return LinearSystem{ΔT}(I+ΔT*ls.A, ΔT*ls.B)
end

# TODO: this should probably know the absolute time (like the `SystemTrajectory`)
struct LTVSystem{ΔT, nx, nu, TD<:AbstractVector{<:LinearSystem{ΔT, nx, nu}}} <: ControlSystem{ΔT, nx, nu}
    "The discrete time series of linear systems."
    dyn::TD
    LTVSystem(dyn::TD) where {ΔT, nx, nu, TD<:AbstractVector{<:LinearSystem{ΔT, nx, nu}}} = begin
        @assert ΔT > 0 "LTVSystem require finite discretization steps."
        new{ΔT, nx, nu, TD}(dyn)
    end
end
LinearizationStyle(::LTVSystem) = TrivialLinearization()
Base.eltype(::Type{<:LTVSystem{ΔT, nx, nu, TD}}) where {ΔT, nx, nu, TD} = eltype(TD)
Base.getindex(ds::LTVSystem, i) = getindex(ds.dyn, i)
Base.setindex!(ds::LTVSystem, v, i) = setindex!(ds.dyn, v, i)
next_x(cs::LTVSystem, xₖ::SVector, uₖ::SVector, k::Int) = next_x(cs.dyn[k], xₖ, uₖ)

# TODO: we could also get away without this type but it seems more clean (?)
struct LTISystem{ΔT,nx,nu,TL<:LinearSystem{ΔT,nx,nu},
                 TXY,TXI}<:ControlSystem{ΔT,nx,nu}
    dyn::TL
    xyids::TXY
    xids::TXI
end
LTISystem(dyn::LinearSystem, xyids) = LTISystem(dyn, xyids, nothing)

LinearizationStyle(::LTISystem) = TrivialLinearization()
xyindex(cs::LTISystem) = cs.xyids
xindex(cs::LTISystem) = cs.xids
Base.eltype(::Type{<:LTISystem{ΔT,nx,nu,TL}}) where {ΔT,nx,nu,TL} = TL
Base.getindex(cs::LTISystem, i) = cs.dyn
next_x(cs::LTISystem, xₖ::SVector, uₖ::SVector, ::Float64) = next_x(cs.dyn, xₖ, uₖ)
