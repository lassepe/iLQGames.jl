"""
$(TYPEDEF)

Abstract representation of a finite horizon game.
"""
abstract type AbstractGame end


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

# TODO add documentation
function uindex end
function horizon end

n_players(g::AbstractGame) = length(uindex(g))

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
time_disc2cont(g::AbstractGame, k::Int, t0::Float64=0.) = (t0 +
                                                           (k-1)*samplingtime(g))
pindex(g) = @S 1:n_players(g)
