import iLQGames: dx, xyindex
using iLQGames:
    ControlSystem, GeneralGame, iLQSolver, PlayerCost, solve, plot_traj,
    FunctionPlayerCost

using StaticArrays

# parametes: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 4, 2, 0.1, 100
# indices of inputs that each player controls
player_inputs = (SVector(1), SVector(2))
# choose initial conditions intial conditions...
x0 = SVector(1, 1, 0, 0.5)

# setup the dynamics
struct Unicycle <: ControlSystem{ΔT,nx,nu} end
# x = (px, py, phi, v)
dx(cs::Unicycle, x, u, t) = SVector(cos(x[3]), x[4]sin(x[3]), u[1], u[2])
xyindex(cs::Unicycle) = tuple(SVector(1,2))
dynamics = Unicycle()

# To setup costs we can derive a custom subtype of `PlayerCost` or simply hand the
# cost function to the FunctionPlayerCost.
costs = (# player-1 wants the unicycle to stay close to the origin
         FunctionPlayerCost{nx,nu}((g, x, u, t) -> (x[1]^2 + x[2]^2 + u[1]^2)),
         # player-2 wants to keep close to 1 m/s
         FunctionPlayerCost{nx,nu}((g, x, u, t) -> ((x[4] - 1)^2 + u[2]^2)))

# the horizon of the game
g = GeneralGame{player_inputs, game_horizon}(dynamics, costs)
# get a solver for the game
solver = iLQSolver(g)
# compute solution (takes about 4ms even though we are using AD)
converged, trajectory, strategies = solve(g, solver, x0)
# ... and plot it
plot_traj(trajectory)
