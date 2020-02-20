"The legacy version of the plot function that computes the player colors form a
simple color map (just distinguishable colors)"
function plot_traj!(plt::Plots.Plot, traj::SystemTrajectory, xy_ids::SIndex,
                    uids::Union{SIndex, Nothing}=nothing,
                    cname::AbstractString="blues", args...; kwargs...)
    # get a color for each player (using offset because first color is too faint)
    player_colors = colormap(cname, length(xy_ids)+1)[2:end]
    # TODO, consider swapping uids and cname globally
    return plot_traj!(plt, traj, xy_ids, player_colors, uids, args...; kwargs...)
end

"Colors each players trajectory by the players cost."
function plot_traj!(plt::Plots.Plot, traj::SystemTrajectory, g::AbstractGame,
                    cs::ColorScheme, rangescale::Union{Tuple, Nothing}=nothing,
                    args...; kwargs...)

    # cost
    pcs = cost(g, traj)
    rangescale = isnothing(rangescale) ? extrema(pcs) : rangescale
    player_colors = [get(cs, pc, rangescale) for pc in pcs]

    return plot_traj!(plt, traj, xyindex(g), player_colors, args...;
                      kwargs...)
end

function plot_traj!(plt::Plots.Plot, traj::SystemTrajectory, g::AbstractGame,
                    player_colors::AbstractArray, args...; kwargs...)
    return  plot_traj!(plt, traj, xyindex(g), player_colors, args...; kwargs...)
end

function plot_traj(traj::SystemTrajectory)
    return plot_traj(traj, [:black], tuple(@S(1:length(eltype(traj.u)))))
end
function plot_traj(traj::SystemTrajectory, player_colors, uids::Union{SIndex, Nothing})
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
function plot_inputs(traj::SystemTrajectory, player_colors, uids::SIndex; kwargs...)
    # names for each input
    nu = length(eltype(traj.u))
    input_labels = reshape(["u$i" for i in 1:nu], 1, nu)
    # find the player color for each input
    input_colors = reshape([player_colors[findfirst(in.(i, uids))] for i in 1:nu], 1, nu)
    return plot(hcat(traj.u...)'; layout=(nu, 1), label=input_labels,
                seriescolor=input_colors, kwargs...)
end

function plot_states(traj::SystemTrajectory)
    nx = length(eltype(traj.x))
    state_labels = reshape(["x$i" for i in 1:nx], 1, nx)
    return plot(hcat(traj.x...)'; layout=(nx, 1), label=state_labels,
                seriescolor=:black)
end

function plot_traj!(plt::Plots.Plot, traj::SystemTrajectory, xy_ids::SIndex,
                    player_colors::AbstractArray,
                    uids::Union{SIndex, Nothing}=nothing, alpha::Float64=1.,
                    legend=:none,
                    path_marker=(:circle, 1, stroke(1, 1., :black)), ; k::Int=1,
                    kp=k)
    # buffer for all the plots

    nu = length(eltype(traj.u))
    pu = plot(; layout=(nu, 1))

    if !isnothing(uids)
        pu = plot_inputs(traj, player_colors, uids)
    end

    for (i, xy_i) in enumerate(xy_ids)
        # the trajectory
        x = collect(x[first(xy_i)] for x in traj.x)
        y = collect(x[last(xy_i)] for x in traj.x)

        plot!(plt, x, y; xlims=(-5, 5), ylims=(-5, 5), seriescolor=player_colors[i],
              seriesalpha=alpha, legend=:none)

        # marker at the current time step
        for k in unique([k, kp])
            if k > 0
                scatter!(plt, [x[k]], [y[k]], seriescolor=player_colors[i],
                         legend=:none)
            end
        end
    end

    return isnothing(uids) ? plt : plot(pu, plt)
end

plot_traj(args...; kwargs...) = begin p = plot(); plot_traj!(p, args...; kwargs...) end

function scatter_positions(args...; kwargs...)
    plt = plot()
    return scatter_positions!(plt, args; kwargs...)
end
function scatter_positions!(plt, x, g, player_colors)
    for (i, xy_i) in enumerate(xyindex(g))
        scatter!(plt, [x[xy_i][1]], [x[xy_i][2]], seriescolor=player_colors[i],
                 markersize=5)
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
