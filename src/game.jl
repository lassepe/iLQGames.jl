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
function lq_approximation end

n_players(g::AbstractGame{uids}) where {uids} = length(uids)
u_idx_ranges(g::AbstractGame{uids}) where {uids} = uids


"--------------------------- Implementations ---------------------------"


"A simple contruction helper that runs some sanity checks on the types"
@inline function game_sanity_checks(uids, TD, TC)
    @assert n_states(TD) == n_states(eltype(TC)) "Cost and dynamics need the same state dimensions."
    @assert n_controls(TD) == n_controls(eltype(TC)) "Cost and dynamics need the same input dimensions"
    @assert eltype(TD) <: LinearSystem "LQGames require linear (time varying) dynamics."
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
struct GeneralGame{uids, TD<:ControlSystem, TC<:StaticVector{<:Any, PlayerCost}} <: AbstractGame{uids}
    dyn::TD
    cost::TC

    GeneralGame{uids}(dyn::TD, cost::TC) where {uids, TD<:ControlSystem,
                                                TC<:StaticVector{<:Any, PlayerCost}} = begin
        game_sanity_checks(uids, TD, TC)
        new{uids, h, TD, TC}(dyn, player_costs)
    end

end

dynamics(g::GeneralGame) = g.dyn
player_costs(g::GeneralGame) = g.cost
lq_approximation(g::GeneralGame) = TODO


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
    player_costs::TC

    LQGame{uids}(dyn::TD, player_costs::TC) where {uids, h, TD<:LTVSystem{h}, TC<:SizedVector{h}} = begin
        game_sanity_checks(uids, TD, eltype(TC))
        new{uids, h, TD, TC}(dyn, player_costs)
    end
end

horizon(g::LQGame{uids, h}) where {uids, h} = h
# TODO: I would really prefer if we did not have to use this!
strategy_type(g::LQGame) = AffineStrategy{n_states(dynamics(g)),
                                          n_controls(dynamics(g)),
                                          SMatrix{n_controls(dynamics(g)),
                                                  n_controls(dynamics(g)),
                                                  Float64,
                                                  n_controls(dynamics(g))*n_states(dynamics(g))},
                                          SVector{n_controls(dynamics(g)),
                                                  Float64}}
dynamics(g::LQGame) = g.dyn
player_costs(g::LQGame) = g.player_costs
lq_approximation(g::LQGame) = g
