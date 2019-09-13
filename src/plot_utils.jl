function gen_fake_data()
    # fake data:
    nx = 5;
    nu = 2;
    N_STEPS = 100;

    A = rand(nx, nx)
    B = rand(nx, nu)

    P = rand(nu, nx)
    α = zeros(nu)
    x0 = rand(nx)

    while any(abs(e) >=1 for e in first(eigen(A+B*P)))
        P = -rand(nu, nx)
        println(maximum(abs.(first(eigen(A+B*P)))))
    end

    xs = [x0]
    us = []
    resize!(xs, N_STEPS)
    resize!(us, N_STEPS-1)
    for k in 1:(N_STEPS-1)
        us[k] = P*xs[k] + α
        xs[k+1] = A*xs[k] + B*us[k]
    end

    return xs, us
end


function plot_traj(traj::SystemTrajectory, xy_ids::SIndex, uids::SIndex; k::Int=1)

    # get a color for each player
    player_colors = distinguishable_colors(length(uids), colorant"darkgreen")
    # buffer for all the plots

    nu = length(eltype(traj.u))
    pu = plot(; layout=(nu, 1))
    pxy = plot()

    # names for each input
    input_labels = reshape(["u$i" for i in 1:nu], 1, nu)
    # find the player color for each input
    input_colors = reshape([player_colors[findfirst(in.(i, uids))] for i in 1:nu], 1, nu)
    plot!(pu, hcat(traj.u...)'; layout=(nu, 1), label=input_labels, seriescolor=input_colors)
    for (i, xy_i) in enumerate(xy_ids)
        # the trajectory
        x = collect(x[first(xy_i)] for x in traj.x)
        y = collect(x[last(xy_i)] for x in traj.x)
        plot!(pxy, x, y; marker=0, xlims=(-5, 5), ylims=(-5, 5), seriescolor=player_colors[i], label="p$i")

        # marker at the current time step
        scatter!(pxy, [x[k]], [y[k]], seriescolor=player_colors[i], label="x_p$i")
    end

    return plot(pu, pxy)
end

# cost plots
function plot_cost(g::AbstractGame, op::SystemTrajectory, dims, i::Int=1,
                   st::Symbol=:contour; k::Int=1)
    lqg = lq_approximation(g, op)
    nx = n_states(dynamics(g))
    nu = n_controls(dynamics(g))
    t = k * sampling_time(dynamics(g))

    offset2vec(Δd1, Δd2) = begin
        Δx = zeros(nx)
        Δx[dims[1]] = Δd1
        if length(dims) == 2
            Δx[dims[2]] = Δd2
        end
        return SVector{nx}(Δx)
    end

    projected_cost(Δd1, Δd2=0) = begin
        Δx = offset2vec(Δd1, Δd2)
        return player_costs(g)[i](op.x[k]+Δx, op.u[k], t)
    end

    projected_cost_approx(Δd1, Δd2=0) = begin
        Δx = offset2vec(Δd1, Δd2)
        c0 = player_costs(g)[i](op.x[k], op.u[k], t)
        return player_costs(lqg)[k][i](Δx, (@SVector zeros(nu))) + c0
    end

    Δd1_range = Δd2_range = -10:0.1:10

    if length(dims) == 1
        p = plot(Δd1_range, projected_cost, label="g")
        plot!(p, Δd1_range, projected_cost_approx, label="lqg")
        return p
    elseif length(dims) == 2
        return plot(Δd1_range, Δd2_range, projected_cost, st=st)
    end

    @assert false "Can only visualize one or two dimensions"
end

function animate_plot(plot_frame::Function, plot_args...;
                      k_range::UnitRange, frame_sample::Int=2, fps::Int=10,
                      filename::String="$(@__DIR__)/../debug_out/test.gif")
    anim = @animate for k in k_range
        plot_frame(plot_args...; k=k)
    end every frame_sample

    return gif(anim, filename; fps=fps)
end

