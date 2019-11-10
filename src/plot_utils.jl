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

""
function plot_traj!(plt::Plots.Plot, traj::SystemTrajectory, xy_ids::SIndex,
                    player_colors::AbstractArray,
                    uids::Union{SIndex, Nothing}=nothing, alpha::Float64=1.,
                    legend::Symbol=:none,
                    path_marker=(:circle, 1, stroke(1, 1., :black)), ; k::Int=1)
    # buffer for all the plots

    nu = length(eltype(traj.u))
    pu = plot(; layout=(nu, 1))

    if !isnothing(uids)
        # names for each input
        input_labels = show_labels ? reshape(["u$i" for i in 1:nu], 1, nu) : []
        # find the player color for each input
        input_colors = reshape([player_colors[findfirst(in.(i, uids))] for i in 1:nu], 1, nu)
        plot!(pu, hcat(traj.u...)'; layout=(nu, 1), label=input_labels,
              seriescolor=input_colors, legend=legend)
    end
    for (i, xy_i) in enumerate(xy_ids)
        # the trajectory
        x = collect(x[first(xy_i)] for x in traj.x)
        y = collect(x[last(xy_i)] for x in traj.x)

        plotargs = (plt, x, y)
        plot!(plotargs...; xlims=(-5, 5), ylims=(-5, 5),
              seriescolor=player_colors[i], label="p$i", legend=legend,
              seriesalpha=alpha)

        # marker at the current time step
        scatter!(plt, [x[k]], [y[k]], seriescolor=player_colors[i],
                 label="x_p$i", legend=legend)
    end

    return isnothing(uids) ? plt : plot(pu, plt, legend=legend)
end

plot_traj(args...; kwargs...) = begin p = plot(); plot_traj!(p, args...; kwargs...) end

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
                      k_range::UnitRange, frame_sample::Int=2, fps::Int=10,
                      filename::String="$(@__DIR__)/../debug_out/test.gif")
    anim = @animate for k in k_range
        plot_frame(plot_args...; k=k)
    end every frame_sample

    return gif(anim, filename; fps=fps)
end
