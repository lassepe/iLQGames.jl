using StaticArrays
using LinearAlgebra
using BenchmarkTools

using iLQGames:
    LinearSystem,
    QuadraticPlayerCost,
    LTVSystem,
    LQGame,
    solve_lq_game!,
    uindex,
    strategytype


function generate_1D_pointmass_game()
    # Testing the solver at a simple example: A two-player point mass 1D system.
    # The state composes of position and oritation. Therefore, the system dynamics
    # are a pure integrator.
    ΔT = 0.1
    H = 10.0
    N_STEPS = Int(H / ΔT)

    # dynamical system
    A = I + ΔT * @SMatrix([0. ΔT;
                           0. 0.])
    B = ΔT * @SMatrix([0.05 0.032;
                       1.0  0.11]);

    dyn = LinearSystem{ΔT}(A, B)
    # costs for each player
    c1 = QuadraticPlayerCost(@SVector([0., 0.]),       # l
                             @SMatrix([1. 0.; 0. 1.]), # Q
                             @SVector([0., 0.]),       # r
                             @SMatrix([1. 0.; 0. 0.])) # R
    c2 = QuadraticPlayerCost(-c1.l,                    # l
                             -c1.Q,                    # Q
                             @SVector([0., 0.]),       # r
                             @SMatrix([0. 0.; 0. 1.])) # R

    costs = @SVector [c1, c2]
    # the lq game (player one has control input 1 and 2; player 2 has control input 3
    ltv_dyn = LTVSystem(SizedVector{N_STEPS}(repeat([dyn], N_STEPS)))
    qtv_costs = SizedVector{N_STEPS}(repeat([costs], N_STEPS))
    lqGame = LQGame{((@SVector [1]), (@SVector [2]))}(ltv_dyn, qtv_costs)

    uindex(lqGame)

    return lqGame
end

function main()
    g1D = generate_1D_pointmass_game()
    strategies1D = strategytype(g1D)(undef)
    b = @benchmark(solve_lq_game!($strategies1D, $g1D), samples=1000)
    display(b)
    solve_lq_game!(strategies1D, g1D)
end

main()
