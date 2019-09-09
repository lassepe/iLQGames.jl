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


function plot_systraj(trajs::Vararg{SystemTrajectory}; xy_ids)

    traj_plots = []

    for traj in trajs
        nu = length(eltype(traj.u))
        pu = plot(; layout=(nu, 1))
        pxy = plot()

        # inputs
        plot!(pu, hcat(traj.u...)'; layout=(nu, 1))
        for xy_i in xy_ids
            x = collect(x[first(xy_i)] for x in traj.x)
            y = collect(x[last(xy_i)] for x in traj.x)
            plot!(pxy, x, y; marker=0, xlims=(-5, 5), ylims=(-2.5, 2.5))
        end

        push!(traj_plots, plot(pu, pxy))
    end

    return plot(traj_plots...; layout=(length(traj_plots), 1))
end
