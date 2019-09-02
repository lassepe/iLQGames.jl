"""
$(TYPEDEF)

Abstract representation of a finite horizon game.
"""
abstract type FiniteHorizonGame{uids, h, nx, nu} end


"""
    $(FUNCTIONNAME)(g::FiniteHorizonGame)

Returns the dynamics of the game, a `ControlSystem`.
"""
function dynamics end

"""
    $(FUNCTIONNAME)(g::FiniteHorizonGame)

Returns the type of the strategy that is a solution to this game.
"""
function strategy_type end

"""
    $(FUNCTIONNAME)(g::FiniteHorizonGame)

Returns the lq approximation of the game.
"""
function lq_approximation end

n_states(g::FiniteHorizonGame{uids, h, nx}) where {uids, h, nx} = nx
n_controls(g::FiniteHorizonGame{uids, h, nx, nu}) where {uids, h, nx, nu} = nu
n_players(g::FiniteHorizonGame{uids}) where {uids} = length(uids)
u_idx_ranges(g::FiniteHorizonGame{uids}) where {uids} = uids
horizon(g::FiniteHorizonGame{uids, h}) where {uids, h} = h

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
- `TP`:   the type of the player costs

# Fields

$(TYPEDFIELDS)
"""
struct FiniteHorizonLQGame{uids, h, nx, nu, TD<:DiscreteTimeVaryingSystem{h}, TP<:SizedVector{h}} <: FiniteHorizonGame{uids, h, nx, nu}
    "The full linear system dynamics. A vector (time) over `LinearSystem`s."
    dyn::TD
    "The cost representation. A vector (time) over vector (player) over
    `QuadraticPlayerCost`"
    player_costs::TP

    FiniteHorizonLQGame{uids}(dyn::TD, player_costs::TP) where {uids, h, TD<:DiscreteTimeVaryingSystem{h}, TP<:SizedVector{h}} = begin
        nx = n_states(TD)
        nu = n_controls(TD)
        @assert isempty(intersect(uids...)) "Invalid uids: Two players can not control the same input"

        @assert sum(length(uis) for uis in uids) == nu "Not all inputs have been assigned to players."
        @assert all(isbits(uir) for uir in uids) "Invalid uids: all ranges should be isbits to make things fast."
        @assert all(eltype(uir) == Int for uir in uids) "Invalid uids: the elements of the u_idx_range should be integers."
        new{uids, h, nx, nu, TD, TP}(dyn, player_costs)
    end
end

# TODO: I would really prefer if we did not have to use this!
strategy_type(g::FiniteHorizonLQGame) = AffineStrategy{n_states(g), n_controls(g),
                                                       SMatrix{n_controls(g), n_controls(g),
                                                               Float64, n_controls(g)*n_states(g)},
                                                       SVector{n_controls(g), Float64}}
lq_approximation(g::FiniteHorizonLQGame) = g
