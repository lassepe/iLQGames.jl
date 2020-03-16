function plot_traj!(plt::Plots.Plot, traj::SystemTrajectory, g::AbstractGame,
                    player_colors::AbstractArray, args...; kwargs...)
    return  plot_traj!(plt, traj, xyindex(g), player_colors, args...; kwargs...)
end

function plot_traj(traj::SystemTrajectory)
    return plot_traj(traj, [:black], tuple(@S(1:length(eltype(traj.u)))))
end
function plot_traj(traj::SystemTrajectory, player_colors::Vector, uids)
    plots = []

    if !isnothing(uids)
        push!(plots, plot_inputs(traj, player_colors, uids))
    end

    push!(plots, plot_states(traj))

    return plot(plots...)
end

function plot_inputs(traj::SystemTrajectory)
    return plot_inputs(traj, [:black], tuple(@S(1:length(eltype(traj.u)))))
end
function plot_inputs(traj::SystemTrajectory, player_colors, uids; kwargs...)
    # names for each input
    nu = length(eltype(traj.u))
    input_labels = reshape([latexstring("u_$i") for i in 1:nu], 1, nu)
    # find the player color for each input
    input_colors = reshape([player_colors[findfirst(in.(i, uids))] for i in 1:nu], 1, nu)
    return plot(hcat(traj.u...)'; layout=(nu, 1),
                legend=:none,
                xlabel="time step",
                ylabel=input_labels,
                seriescolor=input_colors, kwargs...)
end

function plot_states(traj::SystemTrajectory)
    nx = length(eltype(traj.x))
    state_labels = reshape(["x$i" for i in 1:nx], 1, nx)
    return plot(hcat(traj.x...)'; layout=(nx, 1), label=state_labels,
                seriescolor=:black)
end

function plot_traj!(plt::Plots.Plot, traj::SystemTrajectory, xy_ids,
                    player_colors::AbstractArray,
                    uids=nothing,
                    plot_attributes=NamedTuple()
                    ; k::Int=1, kp=nothing)
    # buffer for all the plots
    default_plot_attributes = (legend=:none, seriesalpha=1.,
                               xlabel=L"p_x [m]", ylabel=L"p_y [m]",
                               xlims=(-3.5, 3.5), ylims=(-3.5, 3.5),
                               aspect_ratio=:equal)
    plot_attributes = merge(default_plot_attributes, plot_attributes)

    nu = length(eltype(traj.u))
    pu = plot(; layout=(nu, 1))

    if !isnothing(uids)
        pu = plot_inputs(traj, player_colors, uids)
    end

    if length(xy_ids) == 1
        player_colors = [:black for p in player_colors]
    end

    function mark_position!(p::Plots.Plot, px, py, marker)
        # marker at the current time step
        scatter!(plt, [px], [py]; marker=marker, legend=:none)
    end

    for (i, xy_i) in enumerate(xy_ids)
        # the trajectory
        pxs = collect(x[first(xy_i)] for x in traj.x)
        pys = collect(x[last(xy_i)] for x in traj.x)
        plot!(plt, pxs, pys; seriescolor=player_colors[i], plot_attributes...)
        if k > 0
            mark_position!(plt, pxs[k], pys[k], (:diamond, player_colors[i]))
        end
        if !isnothing(kp) && kp > 0
            mark_position!(plt, pxs[kp], pys[kp], (:circle, player_colors[i]))
        end
    end

    return isnothing(uids) ? plt : plot(pu, plt)
end

plot_traj(args...; kwargs...) = begin p = plot(); plot_traj!(p, args...; kwargs...) end

function scatter_positions(args...; kwargs...)
    plt = plot()
    return scatter_positions!(plt, args; kwargs...)
end
function scatter_positions!(plt, x, g, player_colors, marker=(:circle))
    for (i, xy_i) in enumerate(xyindex(g))
        scatter!(plt, [x[xy_i][1]], [x[xy_i][2]], seriescolor=player_colors[i],
                 marker=marker)
    end
    return plt
end

# cost plots
function plot_cost(g::AbstractGame, op::SystemTrajectory, dims, i::Int=1,
                   st::Symbol=:contour; k::Int=1)
    lqg = lq_approximation(g, op)
    nx = n_states(g)
    nu = n_controls(g)
    t = k * samplingtime(g)

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
        return player_costs(g)[i](g, op.x[k]+Δx, op.u[k], t)
    end

    projected_cost_approx(Δd1, Δd2=0) = begin
        Δx = offset2vec(Δd1, Δd2)
        c0 = player_costs(g)[i](g, op.x[k], op.u[k], t)
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
                      k_range::UnitRange, frame_sample::Int=1, fps::Int=10,
                      filename::String="$(@__DIR__)/../debug_out/test.gif")
    anim = @animate for k in k_range
        plot_frame(plot_args...; k=k)
    end every frame_sample

    return gif(anim, filename; fps=fps)
end
