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


function plot_systraj(trajs::Vararg{SystemTrajectory}; xy_ids, uids)

    # get a color for each player
    player_colors = distinguishable_colors(length(uids), colorant"darkgreen")
    # buffer for all the plots
    traj_plots = []

    for traj in trajs
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
            plot!(pxy, x, y; marker=0, xlims=(-5, 5), ylims=(-2.5, 2.5), seriescolor=player_colors[i], label="p$i")
            # marker for start and end
            scatter!(pxy, [x[1]], [y[1]], seriescolor=player_colors[i], label="x0_p$i")
        end

        push!(traj_plots, plot(pu, pxy))
    end

    return plot(traj_plots...; layout=(length(traj_plots), 1))
end
