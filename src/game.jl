"""
$(TYPEDEF)

Abstract representation of a finite horizon game.
"""
abstract type AbstractGame{uids, h} end


"""
    $(FUNCTIONNAME)(g::AbstractGame)

Returns the dynamics of the game, a `ControlSystem`.
"""
function dynamics end

"""
    $(FUNCTIONNAME)(g::AbstractGame)

Returns the cost representation for the game (a vector of PlayerCost) for each
player.
"""
function player_costs end

"""
    $(TYPEDSIGNATURES)

Returns the type of the strategy that is a solution to this game.
"""
# TODO: I would really prefer if we did not have to use this!
function strategytype(g::AbstractGame)
    elt = AffineStrategy{n_states(g), n_controls(g),
                         SMatrix{n_controls(g), n_states(g), Float64,
                                 n_controls(g)*n_states(g)},
                         SVector{n_controls(g), Float64}}

    return SizedArray{Tuple{horizon(g)}, elt, 1, 1}
end

"""
    $(TYPEDSIGNATURES)

Returns the type of the state that is used in this game.
"""
statetype(g::AbstractGame) = SVector{n_states(g), Float64}

"""
    $(FUNCTIONNAME)(g::AbstractGame)

Returns the lq approximation of the game.
"""
function lq_approximation end

# delegate some function calls to the dynamics
n_states(g::AbstractGame) = n_states(dynamics(g))
n_controls(g::AbstractGame) = n_controls(dynamics(g))
xindex(g::AbstractGame) = xindex(dynamics(g))
xyindex(g::AbstractGame) = xyindex(dynamics(g))
samplingtime(g::AbstractGame) = samplingtime(dynamics(g))

# additional convenience methods
n_players(g::AbstractGame{uids}) where {uids} = length(uids)
uindex(g::AbstractGame{uids}) where {uids} = uids
pindex(g::AbstractGame) = SVector{n_players(g)}(1:n_players(g))
horizon(g::AbstractGame{uids, h}) where {uids, h} = h
time_disc2cont(g::AbstractGame, k::Int, t0::Float64=0.) = (t0 +
                                                           (k-1)*samplingtime(g))

"-------------------------------- Implementations ---------------------------------"


"A simple contruction helper that runs some sanity checks on the types"
@inline function game_sanity_checks(uids, TD, TC)
    @assert(n_states(TD) == n_states(eltype(TC)),
            "Cost and dynamics need the same state dimensions.")
    @assert(n_controls(TD) == n_controls(eltype(TC)),
            "Cost and dynamics need the same input dimensions")
    @assert(isempty(intersect(uids...)),
            "Invalid uids: Two players can not control the same input")
    @assert(sum(length(uis) for uis in uids) == n_controls(TD),
            "Not all inputs have been assigned to players.")
    @assert(all(isbits(uir) for uir in uids),
            "Invalid uids: all ranges should be isbits to make things fast.")
    @assert(all(eltype(uir) == Int for uir in uids),
            "Invalid uids: the elements of the u_idx_range should be integers.")
end


"""
$(TYPEDEF)

A representation of a general game with potentially non-linear dynamics and
non-quadratic costs.
"""
struct GeneralGame{uids, h, TD<:ControlSystem, TC<:StaticVector} <: AbstractGame{uids, h}
    dyn::TD
    cost::TC

    function GeneralGame{uids, h}(dyn::TD, cost::TC) where {uids, h,
                                                            TD<:ControlSystem,
        TC<:StaticVector}
        game_sanity_checks(uids, TD, TC)
        @assert TD <: ControlSystem
        @assert eltype(TC) <: PlayerCost
        new{uids, h, TD, TC}(dyn, cost)
    end
end

dyntype(::Type{<:GeneralGame{uids,h,TD}}) where {uids,h,TD} = TD
dynamics(g::GeneralGame) = g.dyn
player_costs(g::GeneralGame) = g.cost

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
    dyn = SizedVector{h, TLS, 1}(undef)
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

"""
$(TYPEDEF)

A struct to represent a multi-player linear system where each player controls a
range of inputs.

# Parameters

- `uids`: the indices of the control inputs for every player
- `h`:    the horizon of the game (number of steps, Int)
- `nx`    the number of states
- `nu`:   the number of controls
- `TD`:   the type of the dynamics
- `TC`:   the type of the player costs

# Fields

$(TYPEDFIELDS)
"""
struct LQGame{uids, h, TD<:Union{LTVSystem, LTISystem}, TC<:SizedVector{h}} <: AbstractGame{uids, h}
    "The full linear system dynamics. A vector (time) over `LinearSystem`s."
    dyn::TD
    "The cost representation. A vector (time) over vector (player) over
    `QuadraticPlayerCost`"
    pcost::TC

    function LQGame{uids}(dyn::TD, pcost::TC) where {uids,h,TD<:Union{LTVSystem,
                                                                      LTISystem},
                                                     TC<:SizedVector{h}}
        game_sanity_checks(uids, TD, eltype(TC))
        @assert eltype(eltype(TC)) <: QuadraticPlayerCost "LQGames require quadratic cots."
        new{uids, h, TD, TC}(dyn, pcost)
    end
end

function lqgame_preprocess_alloc(g::AbstractGame)
    nx = n_states(g)
    nu = n_controls(g)
    np = n_players(g)
    h = horizon(g)

    # computes the most specific linearization from the knowledge that can be
    # extrcted form g.
    lin_dyn = linearization_alloc(g)

    # costs:
    TQ = SMatrix{nx, nx, Float64, nx*nx}
    TL = SVector{nx, Float64}
    TR = SMatrix{nu, nu, Float64, nu*nu}
    TCi = QuadraticPlayerCost{nx, nu, TQ, TL, TR}
    TC = SVector{np, TCi}
    quad_cost = SizedVector{h, TC, 1}(undef)

    lqg = LQGame{uindex(g)}(lin_dyn, quad_cost)
end

dynamics(g::LQGame) = g.dyn
player_costs(g::LQGame) = g.pcost

function lq_approximation(g::GeneralGame, op::SystemTrajectory)
    lqg = lqgame_preprocess_alloc(g)
    lq_approximation!(lqg, g, op)
    return lqg
end


function lq_approximation!(lqg::LQGame, g::GeneralGame, op::SystemTrajectory)
    for (k, (xₖ, uₖ)) in enumerate(zip(op.x, op.u))
        # discrete linearization along the operating point
        t = time_disc2cont(op, k)
        linearize!(lqg, dynamics(g), xₖ, uₖ, t, k)
        # quadratiation of the cost along the operating point
        player_costs(lqg)[k] = map(player_costs(g)) do pcₖⁱ
            quadraticize(pcₖⁱ, g, xₖ, uₖ, t)
        end
    end
end

@inline function linearize!(lqg::LQGame, dyn::ControlSystem, xₖ, uₖ, t, k::Int)
    return linearize!(LinearizationStyle(dyn), lqg, dyn, xₖ, uₖ, t, k)
end
@inline function linearize!(::LinearizationStyle, lqg, dyn, xₖ, uₖ, t, k::Int)
    return dynamics(lqg)[k] = linearize_discrete(dyn, xₖ, uₖ, t)
end
# TODO: Maybe we can do this less subtle. Currenty we simply do nothing because the
# LQGame already holds the right linear dynamics (created in
# lqgame_preprocess_alloc)
@inline linearize!(::TrivialLinearization, lqg, args...) = dynamics(lqg)
