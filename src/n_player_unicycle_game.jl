struct NPlayerUnicycleCost{TIC,TICR,TSC,TSCR,TPC,TGC}<:NPlayerNavigationCost
    "the index of the player this cost applies to"
    player_id::Int
    inputcost::TIC
    inputconstr::TICR
    statecost::TSC
    stateconstr::TSCR
    proximitycost::TPC
    goalcost::TGC
end

function NPlayerUnicycleCost(player_id, xg, t_final, np;
         inputcost::TIC=QuadCost(SMatrix{2,2}([1. 0.; 0. 1.]) * 10),
         inputconstr::TICR=(SoftConstr(1, deg2rad(-10), deg2rad(10), 50),
                            SoftConstr(2, -9.81, 9.81, 50)),
         statecost::TSC=QuadCost(SMatrix{4,4}(diagm([0., 0., 0., 1.])) * 30),
         stateconstr::TSCR=(SoftConstr(4, -0.05, 2., 50),),
         proximitycost::TPC=ProximityCost(2.0, 50.0, np),
         goalcost::TGC=GoalCost(t_final, xg, SMatrix{4,4}(diagm([1.,1.,1.,0.])) *
                                300)) where {TXI,TUI,TIC,TICR,TSC,TSCR,TPC,TGC}

    return NPlayerUnicycleCost(player_id, inputcost, inputconstr, statecost,
                               stateconstr, proximitycost, goalcost)
end

"-----------------Implementing the NPlayerNavigationCost interface-----------------"

player_id(c::NPlayerUnicycleCost) = c.player_id
inputcost(c::NPlayerUnicycleCost) = c.inputcost
inputconstr(c::NPlayerUnicycleCost) = c.inputconstr
statecost(c::NPlayerUnicycleCost) = c.statecost
stateconstr(c::NPlayerUnicycleCost) = c.stateconstr
proximitycost(c::NPlayerUnicycleCost) = c.proximitycost
goalcost(c::NPlayerUnicycleCost) = c.goalcost
ravoid(g) = mapreduce(c->minimum(proximitycost(c).rs), min, player_costs(g))

"------------------ Implementing Feedback Linearization Interface -----------------"

"Here, we don't transform the cost explicitly but as suggested in
https://arxiv.org/abs/1910.00681 we formulate a new cost directly in ξ coordinates."
function transformed_cost(cs::Unicycle4D, c::NPlayerUnicycleCost, np::Int)
    @unpack t_active, xg = goalcost(c)
    ξg = ξ_from(cs, xg)
    if λ_issingular(cs, ξg)
        @warn "State conversion map is singular at provided goal state."
    end
    return NPlayer2DDoubleIntegratorCost(player_id(c), ξg, t_active, np)
end
