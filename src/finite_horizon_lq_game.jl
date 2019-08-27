using DocStringExtensions
using StaticArrays

abstract type FiniteHorizonGame{uids, h, nx, nu} end

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
struct FiniteHorizonLQGame{uids, h, nx, nu, TD<:StaticVector{h}, TP<:StaticVector{h}} <: FiniteHorizonGame{uids, h, nx, nu}
    "The full linear system dynamics. A vector (time) over `LinearSystem`s."
    dyn::TD
    "The cost representation. A vector (time) over vector (player) over
    `QuadraticPlayerCost`"
    player_costs::TP

    FiniteHorizonLQGame{uids}(dyn::TD, player_costs::TP) where {uids, h, TD<:StaticVector{h}, TP<:StaticVector{h}} = begin
        @assert isempty(intersect(uids...)) "Invalid uids: Two players can not control the same input"
        @assert all(isbits(uir) for uir in uids) "Invalid uids: all ranges should be isbits to make things fast."
        @assert all(eltype(uir) == Int for uir in uids) "Invalid uids: the elements of the u_idx_range should be integers."
        nx = n_states(eltype(TD))
        nu = n_controls(eltype(TD))
        new{uids, h, nx, nu, TD, TP}(dyn, player_costs)
    end
end
