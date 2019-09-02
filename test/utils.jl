using Test

using BenchmarkTools
using Profile
using ProfileView
using InteractiveUtils

# TODO: maybe move this in some other package
macro inferred_with_info(expr)
    quote
        try
            # TODO: maybe remove because doing interpolation here is hard.
            $expr
            @show $(expr.args[1])
            b = @benchmark $expr
            show(stdout, "text/plain", b)
            println("\n")
            @inferred $expr
            true
        catch e
            if typeof(e) == ErrorException && occursin("does not match inferred return type", e.msg)
                @info "Type inference failed. Here is the result of @code_warntype\n"
                @code_warntype $expr
                false
            else
                rethrow(e)
            end
        end
    end
end

macro testset_include(filename)
    @assert filename isa AbstractString
    quote
        @testset $filename begin
            include($filename)
        end;
    end
end

function benchmark_show(f::Function, args...; kwargs...)
    b = @benchmark $f($args..., $kwargs...)
    show(stdout, "text/plain", b)
    println()
end

function profile_show(f::Function, n_samples::Int=1000, args...; kwargs...)
    Profile.clear()
    @profile begin for i in 1:n_samples
            f(args...; kwargs...)
    end end
    ProfileView.view()
end
