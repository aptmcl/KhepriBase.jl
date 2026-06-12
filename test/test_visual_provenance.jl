# test_visual_provenance.jl — golden provenance sidecars + validated minting.
#
# Golden regeneration used to be `cp` + `@test true`: a just-reviewed golden
# and a never-reviewed one were indistinguishable, and minting could never
# refuse a broken artifact.  These tests pin the contract of the provenance
# machinery (Docs/TestingStrategy.md §1a / §8 step 0): sidecar round-trip,
# mint-with-validation, refusal on validator/sanity failure, honest
# "sanity-only" stamping, staleness warnings, and the legacy backfill helper.
using Test
using KhepriBase
import PNGFiles

isdefined(Main, :VisualTests) || include("VisualTests.jl")
using .VisualTests

@testset "golden provenance" begin
  mktempdir() do dir
    let golden_dir = joinpath(dir, "golden"),
        out_dir = mkpath(joinpath(dir, "out")),
        out(name, content) = let p = joinpath(out_dir, name)
          write(p, content)
          p
        end,
        save_png(name, img) = let p = joinpath(out_dir, name)
          PNGFiles.save(p, img)
          p
        end,
        gray = rgba(0.5, 0.5, 0.5, 1.0),
        stamp = (backend="TestBackend", width=1920, height=1080,
                 compare="text_compare")
      mkpath(golden_dir)

      @testset "sidecar round-trip" begin
        let path = VisualTests.write_provenance(
              VisualTests.provenance_path(golden_dir, "demo");
              name="demo", golden="demo.tex", backend_version=nothing,
              validation=(method="sanity:nonempty", passed=true, detail="d"),
              stamp...),
            prov = VisualTests.read_provenance(path)
          @test path == VisualTests.provenance_path(golden_dir, "demo")
          @test prov["schema"] == 1
          @test prov["name"] == "demo"
          @test prov["golden"] == "demo.tex"
          @test prov["backend"] == "TestBackend"
          @test prov["width"] == 1920
          @test prov["height"] == 1080
          @test prov["compare"] == "text_compare"
          @test prov["validation"]["method"] == "sanity:nonempty"
          @test prov["validation"]["passed"] === true
          @test prov["validation"]["detail"] == "d"
          # Auto-filled fields are present and ISO-8601 UTC.
          @test occursin(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", prov["created"])
          @test endswith(prov["created"], "Z")
          @test prov["khepribase"] == string(pkgversion(KhepriBase))
          @test prov["julia"] == string(VERSION)
          # `nothing`-valued fields are omitted, not serialized.
          @test !haskey(prov, "backend_version")
          rm(path)
        end
      end

      @testset "read_provenance on missing and corrupt" begin
        @test VisualTests.read_provenance(joinpath(golden_dir, "nope.provenance.toml")) === nothing
        let path = joinpath(golden_dir, "bad.provenance.toml")
          write(path, "= this is [ not toml")
          @test (@test_logs (:warn, r"corrupt provenance") VisualTests.read_provenance(path)) === nothing
          rm(path)
        end
      end

      @testset "mint_golden! happy path with custom validator" begin
        let test_path = out("scene.tex", "\\draw (0,0) circle (1);"),
            golden_path = joinpath(golden_dir, "scene.tex"),
            (result, validation) = @test_logs VisualTests.mint_golden!(
              "scene", :primitives_2d, test_path, golden_path;
              validate_golden=(n, c, p) -> (true, "oracle:test", "matched analytic measure"),
              stamp...)
          @test result === :minted
          @test validation.method == "oracle:test"
          @test validation.passed
          @test isfile(golden_path)
          @test read(golden_path) == read(test_path)
          let prov = VisualTests.read_provenance(
                VisualTests.provenance_path(golden_dir, "scene"))
            @test prov["golden"] == "scene.tex"
            @test prov["validation"]["method"] == "oracle:test"
            @test prov["validation"]["passed"] === true
            @test prov["validation"]["detail"] == "matched analytic measure"
          end
        end
      end

      @testset "failing validator refuses: no golden, no sidecar" begin
        let test_path = out("badscene.tex", "content"),
            golden_path = joinpath(golden_dir, "badscene.tex"),
            (result, reason) = VisualTests.mint_golden!(
              "badscene", :primitives_2d, test_path, golden_path;
              validate_golden=(n, c, p) -> (false, "oracle:test", "area off by 2x"),
              stamp...)
          @test result === :refused
          @test occursin("oracle:test", reason)
          @test occursin("area off by 2x", reason)
          @test !isfile(golden_path)
          @test !isfile(VisualTests.provenance_path(golden_dir, "badscene"))
        end
      end

      @testset "no validator: sanity-only stamp + mint-time warn" begin
        let test_path = out("plain.tex", "content"),
            golden_path = joinpath(golden_dir, "plain.tex"),
            (result, validation) = @test_logs (:warn, r"without a scene oracle") VisualTests.mint_golden!(
              "plain", :primitives_2d, test_path, golden_path; stamp...)
          @test result === :minted
          @test validation.method == "sanity:nonempty"
          let prov = VisualTests.read_provenance(
                VisualTests.provenance_path(golden_dir, "plain"))
            @test prov["validation"]["method"] == "sanity:nonempty"
            @test prov["validation"]["passed"] === true
          end
        end
        # PNG flavor: a decodable image of the declared size mints as
        # "sanity-only" (pixels were checked for shape, not content).
        let test_path = save_png("pix.png", fill(gray, 3, 4)),
            golden_path = joinpath(golden_dir, "pix.png"),
            (result, validation) = @test_logs (:warn, r"without a scene oracle") VisualTests.mint_golden!(
              "pix", :primitives_2d, test_path, golden_path;
              backend="TestBackend", width=4, height=3,
              compare="pixel_diff_compare")
          @test result === :minted
          @test validation.method == "sanity-only"
        end
      end

      @testset "built-in sanity refusals" begin
        # Empty output never mints.
        let test_path = out("empty.tex", ""),
            golden_path = joinpath(golden_dir, "empty.tex"),
            (result, reason) = VisualTests.mint_golden!(
              "empty", :primitives_2d, test_path, golden_path; stamp...)
          @test result === :refused
          @test occursin("empty", reason)
          @test !isfile(golden_path)
        end
        # A PNG whose dimensions contradict the declared render size never
        # mints — it could never pass the comparison it is the golden for.
        let test_path = save_png("small.png", fill(gray, 5, 5)),
            golden_path = joinpath(golden_dir, "small.png"),
            (result, reason) = VisualTests.mint_golden!(
              "small", :primitives_2d, test_path, golden_path;
              backend="TestBackend", width=1920, height=1080,
              compare="pixel_diff_compare")
          @test result === :refused
          @test occursin("1920", reason)
          @test !isfile(golden_path)
          @test !isfile(VisualTests.provenance_path(golden_dir, "small"))
        end
        # A corrupt PNG refuses cleanly instead of escaping as an ERROR.
        let test_path = out("corrupt.png", "not actually a png"),
            golden_path = joinpath(golden_dir, "corrupt.png"),
            (result, reason) = VisualTests.mint_golden!(
              "corrupt", :primitives_2d, test_path, golden_path;
              backend="TestBackend", width=1920, height=1080,
              compare="pixel_diff_compare")
          @test result === :refused
          @test occursin("decode", reason)
          @test !isfile(golden_path)
          @test !isfile(VisualTests.provenance_path(golden_dir, "corrupt"))
        end
      end

      @testset "warn_if_stale" begin
        # "scene" was minted above with width=1920, height=1080,
        # compare="text_compare".
        @test (@test_logs VisualTests.warn_if_stale(golden_dir, "scene";
                 width=1920, height=1080, compare="text_compare")) === :ok
        @test (@test_logs (:warn, r"may be stale") VisualTests.warn_if_stale(golden_dir, "scene";
                 width=800, height=1080, compare="text_compare")) === :stale
        @test (@test_logs (:warn, r"may be stale") VisualTests.warn_if_stale(golden_dir, "scene";
                 width=1920, height=1080, compare="pixel_diff_compare")) === :stale
        # Sidecar-less goldens are silently :missing — the runner counts them
        # into one summary warning instead of warning per file.
        @test (@test_logs VisualTests.warn_if_stale(golden_dir, "no_sidecar";
                 width=1920, height=1080, compare="text_compare")) === :missing
        # Gensym names from anonymous comparison closures differ across
        # sessions; they opt out of the compare check instead of warning
        # spuriously on every run (size checks still apply).
        @test (@test_logs VisualTests.warn_if_stale(golden_dir, "scene";
                 width=1920, height=1080, compare="#3")) === :ok
        @test (@test_logs (:warn, r"may be stale") VisualTests.warn_if_stale(golden_dir, "scene";
                 width=800, height=1080, compare="#3")) === :stale
      end

      @testset "backfill_provenance" begin
        let legacy_dir = mkpath(joinpath(dir, "legacy_golden"))
          write(joinpath(legacy_dir, "a.tex"), "golden a")
          PNGFiles.save(joinpath(legacy_dir, "b.png"), fill(gray, 2, 2))
          write(joinpath(legacy_dir, "c.aux"), "latex junk")
          write(joinpath(legacy_dir, "c.log"), "latex junk")
          # "a" already has a sidecar: backfill must not touch it.
          VisualTests.write_provenance(
            VisualTests.provenance_path(legacy_dir, "a");
            name="a", golden="a.tex",
            validation=(method="oracle:test", passed=true, detail="d"),
            stamp...)
          let backfilled = backfill_provenance(legacy_dir; backend="TestBackend")
            @test backfilled == ["b"]
            let prov = VisualTests.read_provenance(
                  VisualTests.provenance_path(legacy_dir, "b"))
              @test prov["validation"]["method"] == "legacy-backfill"
              @test prov["validation"]["passed"] === false
              @test prov["compare"] == "pixel_diff_compare"  # inferred from .png
              @test prov["backend"] == "TestBackend"
            end
            # The pre-existing sidecar still records its real validation.
            @test VisualTests.read_provenance(
              VisualTests.provenance_path(legacy_dir, "a"))["validation"]["method"] == "oracle:test"
            # LaTeX build junk gets no sidecar.
            @test !isfile(VisualTests.provenance_path(legacy_dir, "c"))
            # Re-running backfills nothing further.
            @test backfill_provenance(legacy_dir; backend="TestBackend") == String[]
          end
        end
      end
    end
  end
end
