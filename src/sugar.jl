macro S(gen)
    :(SVector{length($(esc(gen)))}($(esc(gen))))
end

macro animated(pfunc, k_range, filename)
    @assert(pfunc.head == :call, "Animated must be used with a functioncall
             that returns a Plots.plot and has a kwarg for the time step `k` to
             render.")

    return :(animate_plot($((esc(a) for a in pfunc.args)...);
                          k_range=$(esc(k_range)),
                          filename=$(esc(filename))))
end
