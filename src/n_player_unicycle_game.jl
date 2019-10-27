struct NPlayerUnicycleCost{nx,nu,xids,uids,TIC,TICR,TSC,TSCR,TPC,TGC} <: NPlayerNavigationCost{nx,nu,xids,uids}
    "the index of the player this cost applies to"
    player_id::Int
    inputcost::TIC
    inputconstr::TICR
    statecost::TSC
    stateconstr::TSCR
    proximitycost::TPC
    goalcost::TGC
end

function NPlayerUnicycleCost{xids,uids}(
        ;player_id, xg, t_final,
         inputcost::TIC=InputCost(SMatrix{2,2}([1. 0.; 0. 1.]) * 10),
         inputconstr::TICR=(SoftConstr(1, deg2rad(-10), deg2rad(10), 50),
                            SoftConstr(2, -9.81, 9.81, 50)),
         statecost::TSC=StateCost(SMatrix{4,4}(diagm([0., 0., 0., 2.])) * 20),
         stateconstr::TSCR=(SoftConstr(4, -0.05, 2., 50),),
         proximitycost::TPC=ProximityCost(1.2, 50),
         goalcost::TGC=GoalCost(t_final, xg, SMatrix{4,4}(diagm([1.,1.,1.,0.])) * 500)) where {xids,uids,TIC,TICR,TSC,TSCR,TPC,TGC}

    nx = sum(length.(xids))
    nu = sum(length.(uids))
    return NPlayerUnicycleCost{nx,nu,xids,uids,TIC,TICR,TSC,TSCR,TPC,TGC}(
        player_id, inputcost, inputconstr, statecost, stateconstr, proximitycost,
        goalcost)
end

"-----------------Implementing the NPlayerNavigationCost interface-----------------"

player_id(c::NPlayerUnicycleCost) = c.player_id
inputcost(c::NPlayerUnicycleCost) = c.inputcost
inputconstr(c::NPlayerUnicycleCost) = c.inputconstr
statecost(c::NPlayerUnicycleCost) = c.statecost
stateconstr(c::NPlayerUnicycleCost) = c.stateconstr
proximitycost(c::NPlayerUnicycleCost) = c.proximitycost
goalcost(c::NPlayerUnicycleCost) = c.goalcost
