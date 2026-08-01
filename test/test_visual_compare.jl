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
        # Earlier failing comparisons used base.png as test output and so
        # already emitted base_diff.png; clear it to observe this case.
        rm(p("base_diff.png"), force=true)
        @test !(@test_logs (:warn, r"dimensions differ") pixel_diff_compare(p("base.png"), p("small.png")))
        # No per-pixel diff exists for mismatched dimensions.
        @test !isfile(p("base_diff.png"))
      end

      @testset "threshold failure emits a saturated diff image" begin
        let result = @test_logs (:warn, r"differ beyond tolerance") pixel_diff_compare(p("base.png"), p("black.png"))
          @test !result
          @test isfile(p("base_diff.png"))
          # gray (0.5) vs black (0) differs by 0.5 on every channel; the
          # 1/max_delta scaling saturates the worst pixel to full white,
          # and alpha is forced opaque.
          let diff = PNGFiles.load(p("base_diff.png")),
              px = convert(KhepriBase.RGBA{Float64}, diff[1, 1])
            @test size(diff) == (10, 10)
            @test px.r ≈ 1.0 && px.g ≈ 1.0 && px.b ≈ 1.0 && px.alpha ≈ 1.0
          end
          rm(p("base_diff.png"))
        end
      end

      @testset "emit_diff=false suppresses the diff image" begin
        @test !(@test_logs (:warn, r"differ beyond tolerance") pixel_diff_compare(p("base.png"), p("black.png"), emit_diff=false))
        @test !isfile(p("base_diff.png"))
      end

      @testset "passing compare writes no diff image" begin
        @test pixel_diff_compare(p("near.png"), p("base.png"))
        @test !isfile(p("near_diff.png"))
      end
    end
  end
end

# Regression: text_compare was a raw read(a) == read(b), which made the text
# goldens architecture-sensitive. The same scene on aarch64 differs by an ulp in
# a sin result, and TikZ's three-decimal printing turns that into 8.506 vs 8.507
# -- reported as a visual regression when nothing had changed. The numeric
# fallback must absorb that while still failing on anything real.
@testset "text_compare" begin
  mktempdir() do dir
    let w = (name, s) -> (q = joinpath(dir, name); write(q, s); q),
        base = "\\fill[s57] (8.506,4.123)--(9.012,4.556)--(8.998,3.221)--cycle;\n",
        a = w("a.tex", base)

      @testset "byte-identical passes" begin
        @test text_compare(a, w("same.tex", base))
      end

      @testset "last-printed-digit drift passes" begin
        @test text_compare(a, w("drift.tex",
          "\\fill[s57] (8.507,4.123)--(9.012,4.557)--(8.997,3.221)--cycle;\n"))
      end

      @testset "full-precision representation drift passes" begin
        @test text_compare(w("p1.pov", "sphere { <0.3, 1.0, 2.5>, 0.5 }\n"),
                           w("p2.pov", "sphere { <0.30000000000000004, 1.0, 2.5>, 0.5 }\n"))
      end

      @testset "a real coordinate change fails" begin
        @test !text_compare(a, w("moved.tex",
          "\\fill[s57] (9.506,4.123)--(9.012,4.556)--(8.998,3.221)--cycle;\n"))
        # just past atol, so the bound is doing work rather than swallowing everything
        @test !text_compare(a, w("edge.tex",
          "\\fill[s57] (8.508,4.123)--(9.012,4.556)--(8.998,3.221)--cycle;\n"))
      end

      @testset "structure must still match exactly" begin
        # same numbers, different command
        @test !text_compare(a, w("cmd.tex",
          "\\draw[s57] (8.506,4.123)--(9.012,4.556)--(8.998,3.221)--cycle;\n"))
        # shade index is a genuine output difference
        @test !text_compare(a, w("shade.tex",
          "\\fill[s58] (8.506,4.123)--(9.012,4.556)--(8.998,3.221)--cycle;\n"))
        # a extra vertex changes the number count
        @test !text_compare(a, w("extra.tex",
          "\\fill[s57] (8.506,4.123)--(9.012,4.556)--(8.998,3.221)--(1.0,1.0)--cycle;\n"))
      end

      @testset "line endings are not silently tolerated" begin
        # keeping the goldens LF everywhere is .gitattributes' job (test/golden/** -text),
        # not something this comparison should paper over
        @test !text_compare(a, w("crlf.tex", replace(base, "\n" => "\r\n")))
      end
    end
  end
end
