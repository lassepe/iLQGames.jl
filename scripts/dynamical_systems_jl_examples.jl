using Revise
using DynamicalSystemsBase
using OrdinaryDiffEq
using Plots
plotlyjs()

# for now let's just see how useful the DynamicalSystems.jl package is

# simple point mass model
#
# Notice: Dynamics notation has u being the full autonomous stat!

@inline @inbounds function point_mass_eom(x, u, t)
    dx_x = x[3]
    dx_y = x[4]
    # using the parameter as a function mappting to the control input
    ddx_x, ddx_y = u(x)

    return SVector{4}(dx_x, dx_y, ddx_x, ddx_y)
end
# For now we can use the ForwardDiff jacobian.
# @inline @inbounds function point_mass_jac(x, u, t)
#     J = @SMatrix [0 1 0 0;
#                   0 0 1 0;
#                   0 0 0 0;
#                   0 0 0 0]
#     return J
# end

# a very simple control policy
p_controller(x) = (1-x[1]-x[3], 1-x[2]-x[4])
zero_controller(x) = (0, 0)

# arguments: equation of motion, initial conditions, control input
@time point_mass_system = ContinuousDynamicalSystem(point_mass_eom, zeros(4), p_controller)

# access the state
@show get_state(point_mass_system)
# get an integrator for the system
@show integ = integrator(point_mass_system; alg=RK4())

## Do some "control stuff with this"
# integrate for some time, setting th e
@info "integrate one Step"
@time step!(integ, 0.1, true)
# notice: this will not change the system state, only the state of the integrator
# but the state of teh integrator has changed

# we can also integrate an entire trajectory:
# horizon
H = 10.0;
# initial state
x0 = [0.0, 0.0, 10.0, 0];
traj = trajectory(point_mass_system, H, x0; alg=RK4())

# plot the the results:
plot!(traj[:, 1], traj[:, 2], xlim=(0,6), ylim=(0,2), title="test")
