
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


function plot_systraj(traj::SystemTrajectory; xy_index::Union{Tuple{Int, Int}, Nothing}=nothing)
    nx = length(eltype(traj.x))
    nu = length(eltype(traj.u))

    # states
    px = plot(hcat(traj.x...)'; layout=(nx, 1))
    # inputs
    pu = plot(hcat(traj.u...)'; layout=(nu, 1))

    if isnothing(xy_index)
        plot(px, pu; layout=2)
    else
        x = collect(x[first(xy_index)] for x in traj.x)
        y = collect(x[last(xy_index)] for x in traj.x)

        pxy = plot(x, y; marker=1)
        lay = @layout [a b; c]
        plot(px, pu, pxy; layout=lay)
    end
end
