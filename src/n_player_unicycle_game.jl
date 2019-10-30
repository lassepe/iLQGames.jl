struct NPlayerUnicycleCost{nx,nu,TXI<:NTuple,TUI<:NTuple,TIC,TICR,TSC,TSCR,TPC,TGC}<:NPlayerNavigationCost{nx,nu}
    "the index of the player this cost applies to"
    player_id::Int
    xids::TXI
    uids::TUI
    inputcost::TIC
    inputconstr::TICR
    statecost::TSC
    stateconstr::TSCR
    proximitycost::TPC
    goalcost::TGC
end

function NPlayerUnicycleCost(player_id, xids::TXI, uids::TUI, xg, t_final;
         inputcost::TIC=InputCost(SMatrix{2,2}([1. 0.; 0. 1.]) * 10),
         inputconstr::TICR=(SoftConstr(1, deg2rad(-10), deg2rad(10), 50),
                            SoftConstr(2, -9.81, 9.81, 50)),
         statecost::TSC=StateCost(SMatrix{4,4}(diagm([0., 0., 0., 2.])) * 20),
         stateconstr::TSCR=(SoftConstr(4, -0.05, 2., 50),),
         proximitycost::TPC=ProximityCost(1.2, 50),
         goalcost::TGC=GoalCost(t_final, xg, SMatrix{4,4}(diagm([1.,1.,1.,0.])) *
                                500)) where {TXI,TUI,TIC,TICR,TSC,TSCR,TPC,TGC}

    nx = sum(length.(xids))
    nu = sum(length.(uids))
    return NPlayerUnicycleCost{nx,nu,TXI,TUI,TIC,TICR,TSC,TSCR,TPC,TGC}(
        player_id, xids, uids, inputcost, inputconstr, statecost, stateconstr,
        proximitycost, goalcost)
end

"-----------------Implementing the NPlayerNavigationCost interface-----------------"

player_id(c::NPlayerUnicycleCost) = c.player_id
inputcost(c::NPlayerUnicycleCost) = c.inputcost
inputconstr(c::NPlayerUnicycleCost) = c.inputconstr
statecost(c::NPlayerUnicycleCost) = c.statecost
stateconstr(c::NPlayerUnicycleCost) = c.stateconstr
proximitycost(c::NPlayerUnicycleCost) = c.proximitycost
goalcost(c::NPlayerUnicycleCost) = c.goalcost
xindex(pc::NPlayerUnicycleCost) = pc.xids
uindex(pc::NPlayerUnicycleCost) = pc.uids
