using Revise
using iLQGames

using StaticArrays
using Plots
pyplot()

car5d = iLQGames.Car5D(1.5)

# very simple controller:
x0 = @SVector [0.0, 0.0, 0.0, 0.0, 1.0]
t0 = 0;
ΔT = 0.01
H = 10

discretization = t0:ΔT:H

traj = zeros(length(discretization), length(x0))
traj[1, :] = x0

u_traj = zeros(length(discretization)-1, 2)

for (i,t) in enumerate(discretization[1:end-1])
    x = SVector{5}(traj[i, :])
    # sinusoidal steering angle, no acceleration
    u = @SVector [cos(t/H * 2π)*0.1, 0.1*sin(t/H * 2π)]
    traj[i+1,:] = iLQGames.integrate(car5d, x, u, t, ΔT)
    u_traj[i,:] = u
end

# TODO figure out how to plot this correctly
p_traj = plot(traj[:, 1], traj[:, 2])
p_states = plot(traj, layout=grid(5, 1))
p_conrols = plot(u_traj)

l = @layout [grid(1, 2); a]
plot(p_traj, p_states, p_conrols, layout=l)
