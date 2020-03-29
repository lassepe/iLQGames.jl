"A simple contruction helper that runs some sanity checks on the types"
@inline function game_sanity_checks(uids, TD, TC)
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
struct GeneralGame{TH<:Integer,TU<:Tuple,TD<:ControlSystem, TC} <: AbstractGame
    h::TH
    uids::TU
    dyn::TD
    cost::TC

    function GeneralGame(h::TH, uids::TU, dyn::TD, cost::TC) where {TH,TU,TD,TC}
        game_sanity_checks(uids, TD, TC)
        return new{TH,TU,TD,TC}(h, uids, dyn, cost)
    end
end

dyntype(::Type{<:GeneralGame{TH,TU,TD}}) where {TH,TU,TD} = TD
dynamics(g::GeneralGame) = g.dyn
player_costs(g::GeneralGame) = g.cost
horizon(g::GeneralGame) = g.h
uindex(g::GeneralGame) = g.uids

"""
    $(TYPEDSIGNATURES)

Transforms a `GeneralGame` `g` with feedback linearizable dynamics (i.e.
`LinearizationStyle(dynamics(g)) isa FeedbackLinearization`) to it's linearized
form.
"""
function transform_to_feedbacklin(g::GeneralGame, x0)
    @assert LinearizationStyle(dynamics(g)) isa FeedbackLinearization
    # transform the dynamics
    lin_dyn = feedbacklin(dynamics(g))
    # approximate the cost by a cost in ξ coordinates
    ξ_cost = map(enumerate(player_costs(g))) do (i, c)
        transformed_cost(dynamics(g), c, n_players(g))
    end |> SVector{n_players(g)}

    # transformed game
    gξ = GeneralGame(horizon(g), uindex(g), lin_dyn, ξ_cost)
    # transform initial conditions
    ξ0 = ξ_from(dynamics(g), x0)
    if λ_issingular(dynamics(g), ξ0)
        @warn "State conversion map is singular at provided initial conditions."
    end

    return gξ, ξ0
end

"""
$(TYPEDEF)

A struct to represent a multi-player linear system where each player controls a
range of inputs.

# Parameters

- `TD`:   the type of the dynamics
- `TC`:   the type of the player costs

# Fields

$(TYPEDFIELDS)
"""
struct LQGame{TU,TD<:Union{LTVSystem, LTISystem}, TC} <: AbstractGame
    uids::TU
    "The full linear system dynamics. A vector (time) of `LinearSystem`s."
    dyn::TD
    "The cost representation. A vector (time) of vector (player) of
    `QuadraticPlayerCost`"
    pcost::TC

    function LQGame(uids::TU, dyn::TD, pcost::TC) where {TU,TD,TC}
        game_sanity_checks(uids, TD, eltype(TC))
        @assert eltype(eltype(TC)) <: QuadraticPlayerCost "LQGames require quadratic costs."
        return new{TU,TD,TC}(uids, dyn, pcost)
    end
end

dynamics(g::LQGame) = g.dyn
player_costs(g::LQGame) = g.pcost
horizon(g::LQGame) = length(g.pcost)
uindex(g::LQGame) = g.uids

function lq_approximation(solver, g::GeneralGame, op::SystemTrajectory)
    lqg = lqgame_preprocess_alloc(g)
    lq_approximation!(lqg, solver, g, op)
    return lqg
end


function lq_approximation!(lqg::LQGame, solver, g::GeneralGame,
                           op::SystemTrajectory)
    for (k, (xₖ, uₖ)) in enumerate(zip(op.x, op.u))
        # discrete linearization along the operating point
        t = time_disc2cont(op, k)
        linearize!(lqg, dynamics(g), xₖ, uₖ, t, k)
        # quadratiation of the cost along the operating point
        player_costs(lqg)[k] = map(player_costs(g)) do pcₖⁱ
            c = quadraticize!(qcache(solver), pcₖⁱ, g, xₖ, uₖ, t)
            return regularize(solver, c)
        end
    end
end

@inline function linearize!(lqg::LQGame, dyn::ControlSystem, xₖ, uₖ, t, k)
    return linearize!(LinearizationStyle(dyn), lqg, dyn, xₖ, uₖ, t, k)
end

@inline function linearize!(::LinearizationStyle, lqg, dyn, xₖ, uₖ, t, k)
    return dynamics(lqg)[k] = linearize_discrete(dyn, xₖ, uₖ, t)
end
# TODO: Maybe we can do this less subtle. Currenty we simply do nothing because the
# LQGame already holds the right linear dynamics (created in
# lqgame_preprocess_alloc)
@inline linearize!(::TrivialLinearization, lqg, args...) = dynamics(lqg)
