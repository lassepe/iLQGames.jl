abstract type NPlayerNavigationCost{nx,nu,xids,uids} <: PlayerCost{nx,nu} end

"The index of the player this cost applies to."
function player_id end
"Returns an `::InputCost`."
function inputcost end
"Returns an iterable of `::SoftConstr` for the inputs."
function inputconstr end
"Returns  a `::StateCost`."
function statecost end
"Returns an interable of `::SoftConstr` for the states."
function stateconstr end
"The `::ProximityCost` for a player with any other player."
function proximitycost end
"The terminal `::GoalCost` to encourage reaching some goal"
function goalcost end

"--------------------------------- Implementation ---------------------------------"

xindex(c::NPlayerNavigationCost{nx, nu, xids}) where {nx, nu, xids} = xids
uindex(c::NPlayerNavigationCost{nx, nu, xids, uids}) where {nx, nu, xids, uids} = uids

@inline function iLQGames.quadraticize(pc::NPlayerNavigationCost, g::GeneralGame,
                                       x::SVector, u::SVector, t::AbstractFloat)
    nx = n_states(pc)
    nu = n_controls(pc)

    xi = xindex(pc)[pc.player_id]
    ui = uindex(pc)[pc.player_id]

    l = @MVector zeros(nx)
    Q = @MMatrix zeros(nx, nx)
    R = @MMatrix zeros(nu, nu)

    # quadratic input cost
    quad!(R, inputcost(pc), ui)
    # input constraints
    for constrᵢ in inputconstr(pc)
        quad!(R, constrᵢ, u, ui)
    end

    # quadratic state cost
    quad!(Q, l, statecost(pc), x[xi], xi)
    # state constraints
    for constrᵢ in stateconstr(pc)
        quad!(Q, l, constrᵢ, x, xi)
    end

    # pairwise proximity cost
    xyi_ego = xyindex(g)[player_id(pc)]
    for (j, xyi_other) in enumerate(xindex(pc))
        j != pc.player_id || continue
        quad!(Q, l, proximitycost(pc), x, xyi_ego, xyi_other)
    end

    # the goal cost
    quad!(Q, l, goalcost(pc), x[xi], xi, t)

    return QuadraticPlayerCost(SMatrix(Q), SVector(l), SMatrix(R))
end

function (pc::NPlayerNavigationCost)(g::AbstractGame, x::SVector, u::SVector, t::Float64)
    # extract the states and inputs for this player
    xi = xindex(pc)[pc.player_id]
    ui = uindex(pc)[pc.player_id]
    xᵢ = x[xi]
    uᵢ = u[ui]
    # setup the cost: each player wan't to:
    cost = 0.0

    # control cost: only cares about own control
    cost += inputcost(pc)(uᵢ)
    for constrᵢ in inputconstr(pc)
        cost += constrᵢ(u, ui)
    end

    # running cost for states (e.g. large steering)
    cost += statecost(pc)(xᵢ)
    for constrᵢ in stateconstr(pc)
        cost += constrᵢ(x, xi)
    end

    # proximity constraint
    # ego positions
    xys = xyindex(g)
    xp_ego = x[xys[player_id(pc)]]
    for (j, xy_other) in enumerate(xys)
        j != player_id(pc) || continue
        # other positions
        xp_other = x[xys[j]]
        cost += proximitycost(pc)(xp_ego, xp_other)
    end

    # goal state cost cost:
    cost += goalcost(pc)(xᵢ, t)

    return cost
end

function generate_nplayer_navigation_game(DynType::Type, CostModelType::Type,
                                          T_horizon::Float64, ΔT::Float64,
                                          goals::Vararg{SVector})
    # the number of players
    np = length(goals)
    # the time at which the goal cost is activated
    t_final = T_horizon - 1.5*ΔT
    h = Int(T_horizon/ΔT)
    # setting up the dynamics
    dyn = ProductSystem(Tuple(DynType{ΔT}() for i in 1:np))

    xids = xindex(dyn)
    uids = uindex(dyn)
    costs = SVector{np}([CostModelType{xids,uids}(player_id=i, xg=goals[i],
                                                  t_final=t_final) for i in 1:np])

    return GeneralGame{uids,h}(dyn, costs)
end