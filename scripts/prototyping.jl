using Revise
using iLQGames

using StaticArrays
using Plots
plotlyjs()

car5d = iLQGames.Car5D(1.5)

# very simple controller:
x0 = @SVector [0.0, 0.0, 0.0, 0.0, 1.0]
t0 = 0;
ΔT = 0.01
H = 10

discretization = t0:ΔT:H

traj = zeros(length(discretization), length(x0))
traj[1, :] = x0

for (i,t) in enumerate(discretization[1:end-1])
    x = SVector{5}(traj[i, :])
    # choose a random control action
    u = @SVector [sin(t/H * π/8), 0]

    traj[i+1,:] = iLQGames.integrate(car5d, x, u, t, ΔT)
end

# TODO figure out how to plot this correctly
p_traj = plot(traj[:, 1], traj[:, 2])
p_states = plot(traj, layout=grid(5, 1))

l = @layout [a b]
plot(p_traj, p_states, layout=l)
