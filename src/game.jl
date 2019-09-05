"""
$(TYPEDEF)

Abstract representation of a finite horizon game.
"""
abstract type AbstractGame{uids} end


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
    $(FUNCTIONNAME)(g::AbstractGame)

Returns the type of the strategy that is a solution to this game.
"""
function strategy_type end

"""
    $(FUNCTIONNAME)(g::AbstractGame)

Returns the lq approximation of the game.
"""
# TODO: which inputs does thes need in general?
function lq_approximation end

n_players(g::AbstractGame{uids}) where {uids} = length(uids)
uindex(g::AbstractGame{uids}) where {uids} = uids


"--------------------------- Implementations ---------------------------"


"A simple contruction helper that runs some sanity checks on the types"
@inline function game_sanity_checks(uids, TD, TC)
    @assert n_states(TD) == n_states(eltype(TC)) "Cost and dynamics need the same state dimensions."
    @assert n_controls(TD) == n_controls(eltype(TC)) "Cost and dynamics need the same input dimensions"
    @assert isempty(intersect(uids...)) "Invalid uids: Two players can not control the same input"
    @assert sum(length(uis) for uis in uids) == n_controls(TD) "Not all inputs have been assigned to players."
    @assert all(isbits(uir) for uir in uids) "Invalid uids: all ranges should be isbits to make things fast."
    @assert all(eltype(uir) == Int for uir in uids) "Invalid uids: the elements of the u_idx_range should be integers."
end


"""
$(TYPEDEF)

A representation of a general game with potentially non-linear dynamics and
non-quadratic costs.
"""
struct GeneralGame{uids, TD<:ControlSystem, TC<:StaticVector} <: AbstractGame{uids}
    dyn::TD
    cost::TC

    GeneralGame{uids}(dyn::TD, cost::TC) where {uids, TD<:ControlSystem,
                                                TC<:StaticVector} = begin
        game_sanity_checks(uids, TD, TC)
        @assert TD <: ControlSystem
        @assert eltype(TC) <: PlayerCost
        new{uids, TD, TC}(dyn, cost)
    end

end

dynamics(g::GeneralGame) = g.dyn
player_costs(g::GeneralGame) = g.cost
# TODO: maybe make a mutable! version of this
function lq_approximation(g::GeneralGame, op::SystemTrajectory)
    # TODO: move this to some lqapproxype(g) or something
    nx = n_states(dynamics(g))
    nu = n_controls(dynamics(g))
    h = length(op.x)
    # preallocate an empty lqgame
    # ltv dynamics
    TA = SMatrix{nx, nx, Float64, nx*nx}
    TB = SMatrix{nx, nu, Float64, nx*nu}
    TLS = LinearSystem{sampling_time(op), nx, nu, TA, TB}
    # time varying dynamics
    dyn = SizedVector{h, TLS}(undef)
    lin_dyn = LTVSystem(dyn)

    # costs:
    TQ = SMatrix{nx, nx, Float64, nx*nx}
    TL = SVector{nx, Float64}
    TR = SMatrix{nu, nu, Float64, nu*nu}
    TCi = QuadraticPlayerCost{nx, nu, TQ, TL, TR}
    TC = SVector{2, TCi}
    quad_cost = SizedVector{h, TC}(undef)


    for (k, (xₖ, uₖ)) in enumerate(zip(op.x, op.u))
        # discrete linearization along the operating point
        # TODO fix later, maybe...
        t = 0.;
        lin_dyn[k] = linearize_discrete(dynamics(g), xₖ, uₖ, t)
        # quadratiation of the cost along the operating point
        quad_cost[k] = map(player_costs(g)) do pcₖⁱ
            quadraticize(pcₖⁱ, xₖ, uₖ, t)
        end
    end

    return LQGame{uindex(g)}(lin_dyn, quad_cost)
end


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
struct LQGame{uids, h, TD<:LTVSystem{h}, TC<:SizedVector{h}} <: AbstractGame{uids}
    "The full linear system dynamics. A vector (time) over `LinearSystem`s."
    dyn::TD
    "The cost representation. A vector (time) over vector (player) over
    `QuadraticPlayerCost`"
    pcost::TC

    LQGame{uids}(dyn::TD, pcost::TC) where {uids, h, TD<:LTVSystem{h}, TC<:SizedVector{h}} = begin
        game_sanity_checks(uids, TD, eltype(TC))
        @assert eltype(TD) <: LinearSystem "LQGames require linear (time varying) dynamics."
        @assert eltype(eltype(TC)) <: QuadraticPlayerCost "LQGames require quadratic cots."
        new{uids, h, TD, TC}(dyn, pcost)
    end
end

horizon(g::LQGame{uids, h}) where {uids, h} = h
# TODO: I would really prefer if we did not have to use this!
strategy_type(g::LQGame) = AffineStrategy{n_states(dynamics(g)),
                                          n_controls(dynamics(g)),
                                          SMatrix{n_controls(dynamics(g)),
                                                  n_states(dynamics(g)),
                                                  Float64,
                                                  n_controls(dynamics(g))*n_states(dynamics(g))},
                                          SVector{n_controls(dynamics(g)),
                                                  Float64}}
dynamics(g::LQGame) = g.dyn
player_costs(g::LQGame) = g.pcost
# TODO this is probably not correct because the zero might still be somewhere else
lq_approximation(g::LQGame) = g
