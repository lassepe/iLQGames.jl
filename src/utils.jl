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


function plot_systraj(xs, us)
    nx = length(first(xs))
    nu = length(first(us))

    # states
    px = plot(hcat(xs...)'; layout=(nx, 1))

    # inputs
    pu = plot(hcat(us...)'; layout=(nu, 1))

    plot(px, pu, layout=2)
end
