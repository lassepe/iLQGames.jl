using Test
using BenchmarkTools

using iLQGames:
    iLQGames,
    dynamics,
    n_states,
    n_controls,
    player_costs,
    quadraticize,
    _quadraticize_ad,
    generate_nplayer_car_game

using StaticArrays
using LinearAlgebra


# generate a game
T_horizon = 10.
ΔT = 0.1
# initial conditions:
# x = (x, y, phi, β, v)
x01 = @SVector [-3., 0., 0., 0., 0.]
x02 = @SVector [0.,  3., -pi/2, 0., 0.]
x0 = vcat(x01, x02)
# goal states (goal position of other player with opposite orientation)
xg1 = @SVector [3., 0., 0., 0., 0.]
xg2 = @SVector [0., -3., -pi/2, 0., 0.]
# generate game
g = generate_nplayer_car_game(T_horizon, ΔT, xg1, xg2)

# unpack for testing
dyn = dynamics(g)
nx = n_states(dyn)
nu = n_controls(dyn)
pc1, pc2 = player_costs(g)

# test quadratization of the cost and quadratization of cost
x = @SVector zeros(nx)
u = @SVector zeros(nu)
t = 0.

@testset "Scaclar comparison old new." begin
    for i in 1:100
        global x = SVector{nx, Float64}(randn(nx))
        global u = SVector{nu, Float64}(randn(nu))
        global t = T_horizon

        for pc in (pc1, pc2)
            c_old = iLQGames._legacy_cost(pc, x, u, t)
            c_new = pc(g, x, u, t)
            @test isapprox(c_old, c_new)
        end
    end
end;

@testset "Quadratization comparison" begin
    for i in 1:1000
        global x = SVector{nx, Float64}(randn(nx))
        global u = SVector{nu, Float64}(randn(nu))
        global t = T_horizon

        for pc in (pc1, pc2)
            qc_manual = quadraticize(pc, g, x, u, t)
            qc_ad= _quadraticize_ad(pc, g, x, u, t)

            @test isapprox(qc_manual.Q, qc_ad.Q)
            @test isapprox(qc_manual.l, qc_ad.l)
            @test isapprox(qc_manual.R, qc_ad.R)
        end
    end
end;
