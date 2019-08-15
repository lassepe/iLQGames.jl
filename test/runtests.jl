# Some Simple Tests
using Test
using iLQGames
using BenchmarkTools

# TODO: maybe move this in some other package
macro inferred_with_info(expr)
    quote
        try
            # TODO: maybe remove because doing interpolation here is hard.
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

@testset "control_system.jl" begin
    include("test_control_system.jl")
end
