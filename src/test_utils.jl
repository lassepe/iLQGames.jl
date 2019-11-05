module TestUtils
    using Test
    using StaticArrays

    using iLQGames:
        GeneralGame,
        QuadCache,
        n_states,
        n_controls,
        quadraticize!,
        _quadraticize_ad,
        player_costs

    export
        quad_sanity_check

    # TODO: move to general game test utils
    function quad_sanity_check(g::GeneralGame)
        nx = n_states(g)
        nu = n_controls(g)
        qcache = zero(QuadCache{n_states(g), n_controls(g)})
        @testset "Quadratization Sanity Check" begin
            for i in 1:100
                x = SVector{nx, Float64}(randn(nx))
                u = SVector{nu, Float64}(randn(nu))
                t = 10.0

                for pc in player_costs(g)
                    qc_manual = quadraticize!(qcache, pc, g, x, u, t)
                    qc_ad = _quadraticize_ad(pc, g, x, u, t)

                    @test isapprox(qc_manual.Q, qc_ad.Q)
                    @test isapprox(qc_manual.l, qc_ad.l)
                    @test isapprox(qc_manual.R, qc_ad.R)
                end
            end
        end;
    end
end
