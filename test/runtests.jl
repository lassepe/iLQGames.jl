# Some Simple Tests
using Test
using InteractiveUtils
using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.samples=1
BenchmarkTools.DEFAULT_PARAMETERS.evals=1

macro testset_include(filename)
    @assert filename isa AbstractString
    quote
        @testset $filename begin
            @info $filename
            include($filename)
        end;
    end
end

macro inferred_with_info(expr)
    rexpr = quote
        try
            $expr
            @show $(expr.args[1])
            @inferred $expr
            true
        catch e
            if typeof(e) == ErrorException && occursin("does not match inferred return type", e.msg)
                @info "Type inference failed. Here is the result of @code_warntype\n"
                InteractiveUtils.@code_warntype $expr
                false
            else
                rethrow(e)
            end
        end
    end
    return esc(rexpr)
end # macro


@testset "all" begin
    @testset_include "test_control_system.jl"
    @testset_include "test_linear_system.jl"
    @testset_include "test_product_system.jl"
    @testset_include "test_system_trajectory.jl"
    @testset_include "test_solve_lq_game.jl"
    @testset_include "test_ilq_solver.jl"
    @testset_include "test_nplayer_navigation.jl"
    @testset_include "test_flat.jl"
end;
