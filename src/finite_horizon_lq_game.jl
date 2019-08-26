using DocStringExtensions
using StaticArrays

"""
$(TYPEDEF)

A struct to represent a multi-player linear system where each player controls a
range of inputs.

# Parameters

- `uids`: the indices of the control inputs for every player
- `np`:   the number of players
- `nx`    the number of states
- `nu`:   the number of controls
- `h`:    the horizon of the game (number of steps, Int)

# Fields

$(TYPEDFIELDS)
"""
struct FiniteHorizonLQGame{uids, h, TD<:StaticVector{h}, TP<:StaticVector{h}}
    "The full linear system dynamics. A vector (time) over `LinearSystem`s."
    dyn::TD
    "The cost representation. A vector (time) over vector (player) over
    `QuadraticPlayerCost`"
    player_costs::TP

    FiniteHorizonLQGame{uids}(dyn::TD, player_costs::TP) where {uids, h, TD<:StaticVector{h}, TP<:StaticVector{h}} = begin
        @assert isempty(intersect(uids...)) "Invalid uids: Two players can not control the same input"
        @assert all(isbits(uir) for uir in uids) "Invalid uids: all ranges should be isbits to make things fast."
        @assert all(eltype(uir) == Int for uir in uids) "Invalid uids: the elements of the u_idx_range should be integers."
        new{uids, h, TD, TP}(dyn, player_costs)
    end
end

n_states(g::FiniteHorizonLQGame{uids, h, TD}) where {uids, h, TD} = n_states(eltype(TD))
n_controls(g::FiniteHorizonLQGame{uids, h, TD}) where {uids, h, TD} = n_controls(eltype(TD))
n_players(g::FiniteHorizonLQGame{uids}) where {uids} = length(uids)
u_idx_ranges(g::FiniteHorizonLQGame{uids}) where {uids} = uids
horizon(g::FiniteHorizonLQGame{uids, h}) where {uids, h} = h
