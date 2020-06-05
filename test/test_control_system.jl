using Test

using StaticArrays
using iLQGames:
    ControlSystem,
    Lorenz3D,
    dx,
    integrate,
    linearize

l3 = Lorenz3D{0.1}(10., 28., 8//3)

# test some linearization
@testset "control_system.jl" begin
    @test @inferred_with_info integrate(l3, @SVector([0., 0., 0.]), @SVector([0., 1.]), 0., 0.1)
    @test @inferred_with_info dx(l3, @SVector([2., 1., 7.]), @SVector([0., 1.]), 0.)
    @test @inferred_with_info linearize(l3, @SVector([2., 1., 7.]), @SVector([0., 1.]), 0.)
end;
