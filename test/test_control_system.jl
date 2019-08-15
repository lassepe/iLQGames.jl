using StaticArrays

l3 = iLQGames.Lorenz3D(10., 28., 8//3)

# test some linearization
@testset "Type inference tests" begin
    @test @inferred_with_info iLQGames.dx(l3, @SVector([2, 1, 7]), @SVector([0, 1]), 0)
    @test @inferred_with_info iLQGames.linearize(l3, @SVector([2, 1, 7]), @SVector([0, 1]), 0)
end
