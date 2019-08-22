using Test
using StaticArrays

using iLQGames:
    LinearSystem,
    QuadraticPlayerCost,
    FiniteHorizonLQGame,
    solve_lq_game,
    n_states,
    n_controls,
    n_players,
    u_idx_ranges,
    horizon

using Profile
using ProfileView
using BenchmarkTools


# solve two-player inifinite horizion (time-invariant) LQ game by Lyapunov
# iterations
# TODO this could also use the struct version
function solve_lyapunov_iterations(A::AbstractArray, B1::AbstractArray, B2::AbstractArray,
                                   Q1::AbstractArray, Q2::AbstractArray,
                                   R11::AbstractArray, R12::AbstractArray,
                                   R21::AbstractArray, R22::AbstractArray, n_iter::Int=100)

    # Initialize cost-to-go with terminal state cost
    Z1 = Q1
    Z2 = Q2

    # Initialize controls
    P1 = (R11 + B1' * Z1 * B1) \ (B1' * Z1 * A)
    P2 = (R22 + B2' * Z2 * B2) \ (B2' * Z2 * A)

    for ii in 1:n_iter
        P1_old = P1
        P2_old = P2

        P1 = (R11 + B1' * Z1 * B1) \ (B1' * Z1 * (A - B2 * P2_old))
        P2 = (R22 + B2' * Z2 * B2) \ (B2' * Z2 * (A - B1 * P1_old))

        Z1 = (A - B1 * P1 - B2 * P2)' * Z1 * (A - B1 * P1 - B2 * P2) + P1' * R11 * P1 + P2' * R12 * P2 + Q1
        Z2 = (A - B1 * P1 - B2 * P2)' * Z2 * (A - B1 * P1 - B2 * P2) + P1' * R21 * P1 + P2' * R22 * P2 + Q2
    end

    return (P1, P2)
end

# Testing the solver at a simple example: A two-player point mass 1D system.
# The state composes of position and oritation. Therefore, the system dynamics
# are a pure integrator.
const ΔT = 0.1
const H = 10.0
const N_STEPS = Int(H / ΔT)

# dynamical system
A = SMatrix{2, 2}([1. ΔT; 0. 1.])
B = SMatrix{2, 2}([0.5*ΔT^2 ΔT; 0.32*ΔT^2 0.11*ΔT]')
dyn = LinearSystem(A, B)
# costs for each player
c1 = QuadraticPlayerCost((@SMatrix [1. 0.; 0. 1.]), # Q
                         (@SVector [0., 0.]), # l
                         (@SMatrix [1. 0.; 0. 0.])) # R
c2 = QuadraticPlayerCost(-c1.Q,                     # Q
                         -c1.l,                     # l
                         (@SMatrix [0. 0.; 0. 1.])) # R

costs = @SVector [c1, c2]
# the lq game (player one has control input 1 and 2; player 2 has control input 3
const N_STEPS = 100
const ltv_dyn = SVector{N_STEPS}(repeat([dyn], N_STEPS))
const qtv_costs = SVector{N_STEPS}(repeat([costs], N_STEPS))
const lqGame = FiniteHorizonLQGame{((@SVector [1]), (@SVector [2]))}(ltv_dyn, qtv_costs)

# test all the function calls:
using Test
@test n_states(lqGame) == size(A)[1]
@test n_controls(lqGame) == size(B)[2]
@test n_players(lqGame) == length(costs)
@test horizon(lqGame) == N_STEPS
u_idx_ranges(lqGame)

const A_disc = SMatrix{2, 2}([1.0 ΔT; 0.0 1.0])
const B1_disc = SMatrix{2, 1}([0.5 * ΔT^2, ΔT])
const B2_disc = SMatrix{2, 1}([0.32 * ΔT^2, 0.11 * ΔT])

# state cost
const Q1 = @SMatrix [1. 0.; 0. 1.]
const Q2 = -Q1
const l1 = @SVector zeros(2)
const l2 = -l1

# control cost
const R11 = @SMatrix [1.]
const R12 = @SMatrix [0.]
const R21 = @SMatrix [0.]
const R22 = @SMatrix [1.]
# the matrix version of the problem
const As = SVector{N_STEPS}(repeat([A_disc], N_STEPS))
const Bs = SVector{N_STEPS}(repeat([SVector{2}([B1_disc, B2_disc])], N_STEPS))
const Qs = SVector{N_STEPS}(repeat([SVector{2}([Q1, Q2])], N_STEPS))
const ls = SVector{N_STEPS}(repeat([SVector{2}([l1, l2])], N_STEPS))
const Rs = SVector{N_STEPS}(repeat([SVector{2}([SVector{2}([R11, R12]), SVector{2}([R21, R22])])], N_STEPS))

function benchmark_solve_lq_game()
    b = @benchmark solve_lq_game($lqGame)
    show(stdout, "text/plain", b)
    println()
end

function profile_solve_lq_game()
    Profile.clear()
    @profile begin for i in 1:1000
            solve_lq_game(lqGame)
    end end
    ProfileView.view()
end


function test_lyapunov()
    # 1. Lyapunov solution for the inifnite horizion problem
    P1_lyap, P2_lyap = solve_lyapunov_iterations(A_disc, B1_disc, B2_disc, Q1, Q2, R11, R12, R21, R22)
    P_lyap = [P1_lyap; P2_lyap]

    # 2. LQ game solution
    strategies = solve_lq_game(lqGame)
    P_lqg, _ = first(strategies)

    @info """

    P_lyap: $(P_lyap)
    P_lqg:  $(P_lqg)
    """

    @testset "lyapunov" begin
        @test isapprox(P_lyap, P_lqg)
    end;
end

function test_global_nash()
    # compute nash solution:
    strategies = solve_lq_game(lqGame)
    (P1_lqg, P2_lqg), (α1_lqg, α2_lqg) = first(strategies)
    # TODO -- for now we skil this test
end

@testset "solve_lq_games.jl" begin
    # benchmark_solve_lq_game()
    test_lyapunov()
    # test_global_nash()
end;

