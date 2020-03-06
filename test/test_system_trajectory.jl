using Test

using iLQGames: SystemTrajectory, time_cont2disc, timepoints, zero!

h, ΔT, nx, nu = 100, 0.1, 4, 2
traj = Base.zero(SystemTrajectory{h, ΔT, nx, nu})

@testset "time" begin
    @test time_cont2disc(traj, 5.0) ==  51
    @test collect(timepoints(traj)) ≈ collect(0:0.1:9.9)
end

@testset "zero" begin
    traj_pertubed = deepcopy(traj)
    for i in eachindex(traj_pertubed.x)
        traj_pertubed.x[i] = randn(typeof(traj_pertubed.x[i]))
        traj_pertubed.u[i] = randn(typeof(traj_pertubed.u[i]))
    end
    @assert any(!iszero(x) for x in traj_pertubed.x)
    @assert any(!iszero(u) for u in traj_pertubed.u)

    function test_all_zero(traj)
        @test all(iszero(x) for x in zero(traj.x))
        @test all(iszero(u) for u in zero(traj.u))
    end

    test_all_zero(traj)
    test_all_zero(zero(traj_pertubed))
    zero!(traj_pertubed)
    test_all_zero(traj_pertubed)
end

