struct QuadCache{nx,nu,TL<:MVector{nx},TQ<:MMatrix{nx,nx},
                 TRL<:MVector{nu},TRQ<:MMatrix{nu,nu}}
    l::TL
    Q::TQ
    r::TRL
    R::TRQ
end

function Base.zero(::Type{<:QuadCache{nx,nu}}) where {nx,nu}
    return QuadCache(@MVector(zeros(nx)), @MMatrix(zeros(nx,nx)),
                     @MVector(zeros(nu)), @MMatrix(zeros(nu,nu)))
end

function reset!(qcache::QuadCache)
    fill!(qcache.l, 0.)
    fill!(qcache.Q, 0.)
    fill!(qcache.r, 0.)
    fill!(qcache.R, 0.)
end

"""
    $(TYPEDSIGNATURES)

Returns the most specific preallocation for the linearization.
"""
function linearization_alloc(g::AbstractGame)
    linearization_alloc(LinearizationStyle(dynamics(g)), g)
end
function linearization_alloc(::JacobianLinearization, g::AbstractGame)
    nx = n_states(g)
    nu = n_controls(g)
    ΔT = samplingtime(g)
    h = horizon(g)
    # preallocate an empty lqgame
    # ltv dynamics
    TA = SMatrix{nx, nx, Float64, nx*nx}
    TB = SMatrix{nx, nu, Float64, nx*nu}
    TLS = LinearSystem{ΔT, nx, nu, TA, TB}
    dyn = SizedVector{h, TLS}(undef)
    # time varying dynamics
    return LTVSystem(dyn)
end
function linearization_alloc(::FeedbackLinearization, g::AbstractGame)
    @info("""
          Processing game with feedback linearizable dynamics. Consider
          transforming the game before feeding to the game solver. The system will
          *NOT* be implicitly feedback linearized.
          """, maxlog=1)
    return linearization_alloc(JacobianLinearization(), g::AbstractGame)
end
linearization_alloc(::TrivialLinearization, g::AbstractGame) = dynamics(g)

function lqgame_preprocess_alloc(g::AbstractGame)
    nx = n_states(g)
    nu = n_controls(g)
    np = n_players(g)
    h = horizon(g)

    # computes the most specific linearization from the knowledge that can be
    # extrcted form g.
    lin_dyn = linearization_alloc(g)

    # costs:
    TL = SVector{nx, Float64}
    TQ = SMatrix{nx, nx, Float64, nx*nx}
    TRL = SVector{nu, Float64}
    TRQ = SMatrix{nu, nu, Float64, nu*nu}
    TCi = QuadraticPlayerCost{nx, nu, TL, TQ, TRL, TRQ}
    TC = SVector{np, TCi}
    quad_cost = SizedVector{h, TC}(undef)

    lqg = LQGame(uindex(g), lin_dyn, quad_cost)
end

