# test_visual_compare.jl — pixel_diff_compare unit tests.
#
# Regression: the tolerance path used to be dead code — pixel_diff_compare
# fetched PNGFiles from Base.loaded_modules inside a bare try/catch returning
# false, and no consumer ever loaded PNGFiles, so every comparison silently
# degraded to byte equality with zero diagnostics.
using Test
using KhepriBase
import PNGFiles

include("VisualTests.jl")
using .VisualTests

@testset "pixel_diff_compare" begin
  mktempdir() do dir
    let gray = rgba(0.5, 0.5, 0.5, 1.0),
        near_gray = rgba(0.51, 0.51, 0.51, 1.0),  # Δ ≈ 0.008 < atol 0.02
        black = rgba(0.0, 0.0, 0.0, 1.0),
        base = fill(gray, 10, 10),
        p(name) = joinpath(dir, name)
      PNGFiles.save(p("base.png"), base)

      @testset "PNGFiles loads on demand" begin
        @test VisualTests.load_pngfiles() isa Module
      end

      @testset "byte-identical images pass" begin
        cp(p("base.png"), p("copy.png"))
        @test pixel_diff_compare(p("base.png"), p("copy.png"))
      end

      @testset "sub-atol pixel noise passes despite different bytes" begin
        PNGFiles.save(p("near.png"), fill(near_gray, 10, 10))
        @test read(p("near.png")) != read(p("base.png"))
        @test pixel_diff_compare(p("base.png"), p("near.png"))
      end

      @testset "gross difference fails with diagnostics" begin
        PNGFiles.save(p("black.png"), fill(black, 10, 10))
        @test !(@test_logs (:warn, r"differ beyond tolerance") pixel_diff_compare(p("base.png"), p("black.png")))
      end

      @testset "threshold gates the fraction of differing pixels" begin
        let one_bad = copy(base)
          one_bad[1, 1] = black
          PNGFiles.save(p("one_bad.png"), one_bad)
          # 1 differing pixel out of 100 = exactly the default 1% threshold.
          @test pixel_diff_compare(p("base.png"), p("one_bad.png"))
          @test !(@test_logs (:warn, r"differ beyond tolerance") pixel_diff_compare(p("base.png"), p("one_bad.png"), threshold=0.005))
        end
      end

      @testset "atol gates the per-channel delta" begin
        # The same near-gray image fails once the per-channel gate is
        # tighter than the 0.008 quantized delta.
        @test !(@test_logs (:warn, r"differ beyond tolerance") pixel_diff_compare(p("base.png"), p("near.png"), atol=0.001))
      end

      @testset "dimension mismatch fails loudly" begin
        PNGFiles.save(p("small.png"), fill(gray, 5, 5))
        @test !(@test_logs (:warn, r"dimensions differ") pixel_diff_compare(p("base.png"), p("small.png")))
      end
    end
  end
end
