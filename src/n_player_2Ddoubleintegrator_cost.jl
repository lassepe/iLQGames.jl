# TODO: generalize to K-integrator setting
struct NPlayer2DDoubleIntegratorCost{nx,nu,TXI<:NTuple,TUI<:NTuple,TIC,TICR,TSC,
                                     TSCR,TPC,TGC} <: NPlayerNavigationCost{nx,nu}
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

function NPlayer2DDoubleIntegratorCost(player_id, xids::TXI, uids::TUI, xg, t_final;
         inputcost::TIC=QuadCost(SMatrix{2,2}([1. 0.; 0. 1.]) * 10),
         inputconstr::TICR=tuple(),
         statecost::TSC=QuadCost(SMatrix{4,4}(diagm([0., 1., 0., 1.])) * 50),
         # TODO: think of a way to enforce v > 0
         stateconstr::TSCR=(SoftConstr(2, -2., 2., 50),
                            SoftConstr(4, -2., 2., 50)),
         proximitycost::TPC=ProximityCost(1.2, 50),
         goalcost::TGC=GoalCost(t_final, xg, SMatrix{4,4}(diagm([1.,0.,1.,0.])) *
                                300)) where {TXI,TUI,TIC,TICR,TSC,TSCR,TPC,TGC}

    nx = sum(length.(xids))
    nu = sum(length.(uids))
    return NPlayer2DDoubleIntegratorCost{nx,nu,TXI,TUI,TIC,TICR,TSC,TSCR,TPC,TGC}(
        player_id, xids, uids, inputcost, inputconstr, statecost, stateconstr,
        proximitycost, goalcost)
end

"-----------------Implementing the NPlayerNavigationCost interface-----------------"

player_id(c::NPlayer2DDoubleIntegratorCost) = c.player_id
inputcost(c::NPlayer2DDoubleIntegratorCost) = c.inputcost
inputconstr(c::NPlayer2DDoubleIntegratorCost) = c.inputconstr
statecost(c::NPlayer2DDoubleIntegratorCost) = c.statecost
stateconstr(c::NPlayer2DDoubleIntegratorCost) = c.stateconstr
proximitycost(c::NPlayer2DDoubleIntegratorCost) = c.proximitycost
goalcost(c::NPlayer2DDoubleIntegratorCost) = c.goalcost
xindex(pc::NPlayer2DDoubleIntegratorCost) = pc.xids
uindex(pc::NPlayer2DDoubleIntegratorCost) = pc.uids
