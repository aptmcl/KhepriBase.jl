# test_summary.jl — Headless tests for the model summary / comparison layer (src/Summary.jl).
#
# Models are small NamedTuples mirroring the shape produced by introspect_model / the
# bim_model helper in test_codegen.jl, built on the MockBackend under with_introspection
# (proxies only — nothing is realized). Summaries must roundtrip through
# write_summary/read_summary and compare_summaries must implement the PASS/FAIL/WARN
# contract, including the :template_levels and :stair_railings allowances.

using Test
using KhepriBase

# Summary.jl is not yet wired into KhepriBase.jl; load it into the module when absent.
isdefined(KhepriBase, :model_summary) ||
  Base.include(KhepriBase, joinpath(@__DIR__, "..", "src", "Summary.jl"))
using KhepriBase: model_summary, summary_string, write_summary, read_summary,
                  compare_summaries

include(joinpath(@__DIR__, "TestMockBackend.jl"))

# Mirrors the bim_model helper in test_codegen.jl (empty slots default to []).
summary_bim_model(; levels = [], walls = [], floors = [], columns = [], beams = [],
                    ceilings = [], roofs = [], fixtures = [], stairs = [], railings = [],
                    groups = [], family_meta = IdDict{Any, FamilyMeta}()) =
  (levels = levels, walls = walls, floors = floors, columns = columns, beams = beams,
   ceilings = ceilings, roofs = roofs, fixtures = fixtures, stairs = stairs,
   railings = railings, groups = groups, family_meta = family_meta)

# 2-level rectangular 6x4 room: 4 walls (1 door + 1 window on wall 1), 3 columns, 1 slab.
sample_model() =
  let l0 = level(0.0), l1 = level(3.0),
      corners = [xyz(0, 0, 0), xyz(6, 0, 0), xyz(6, 4, 0), xyz(0, 4, 0)],
      ws = [wall(open_polygonal_path([corners[i], corners[mod1(i + 1, 4)]]),
                 bottom_level = l0, top_level = l1) for i in 1:4],
      cols = [column(xyz(2.0 * i, 0.0, 0.0), bottom_level = l0, top_level = l1)
              for i in 0:2],
      slabs = [slab(closed_polygonal_path(corners), level = l0)]
    push!(ws[1].doors, door(ws[1], xy(1.0, 0.0)))
    push!(ws[1].windows, window(ws[1], xy(3.0, 0.9)))
    summary_bim_model(levels = [l0, l1], walls = ws, floors = slabs, columns = cols)
  end

fail_lines(r) = filter(startswith("FAIL "), r.lines)
warn_lines(r) = filter(startswith("WARN "), r.lines)

@testset "model summary layer" begin
  with_mock_backend() do b
    with_introspection(b) do

      @testset "counts, lengths, areas, bbox" begin
        let s = model_summary(sample_model())
          @test s["count.levels"] == 2
          @test s["count.walls"] == 4
          @test s["count.curtain_walls"] == 0
          @test s["count.floors"] == 1
          @test s["count.columns"] == 3
          @test s["count.doors"] == 1
          @test s["count.windows"] == 1
          @test s["count.beams"] == 0
          @test s["count.groups"] == 0
          @test s["count.fallback_meshes"] == 0
          @test s["level_elevations"] == [0.0, 3.0]
          @test s["total_wall_length"] == 20.0
          @test s["total_slab_area"] == 24.0
          @test s["bbox"] == (0.0, 0.0, 0.0, 6.0, 4.0, 0.0)
        end
      end

      @testset "curtain walls counted separately, excluded from wall length" begin
        let m = sample_model(),
            cw = curtain_wall(open_polygonal_path([xyz(0, 8, 0), xyz(6, 8, 0)]),
                              bottom_level = level(0.0), top_level = level(3.0)),
            s = model_summary(summary_bim_model(levels = m.levels,
                                                walls = vcat(m.walls, [cw]),
                                                floors = m.floors, columns = m.columns))
          @test s["count.walls"] == 4
          @test s["count.curtain_walls"] == 1
          @test s["total_wall_length"] == 20.0     # curtain wall length not included
          @test s["bbox"][5] == 8.0                # ...but its vertices extend the bbox
        end
      end

      @testset "slab with hole subtracts hole area" begin
        let outer = closed_polygonal_path([xyz(0, 0, 0), xyz(10, 0, 0), xyz(10, 10, 0), xyz(0, 10, 0)]),
            hole = closed_polygonal_path([xyz(2, 2, 0), xyz(4, 2, 0), xyz(4, 4, 0), xyz(2, 4, 0)]),
            s = model_summary(summary_bim_model(levels = [level(0.0)],
                                                floors = [slab(region(outer, hole), level = level(0.0))]))
          @test s["total_slab_area"] == 96.0       # 100 - 4
        end
      end

      @testset "groups and group instances" begin
        let g1 = (type_name = "G1", members = [1, 2], instances = [xyz(0, 0, 0), xyz(5, 0, 0)]),
            g2 = (type_name = "G2", members = [3], instances = [xyz(9, 9, 0)]),
            s = model_summary(summary_bim_model(groups = [g1, g2]))
          @test s["count.groups"] == 2
          @test s["count.group_instances"] == 3
        end
      end

      @testset "fixtures and fallback meshes via extended NamedTuple" begin
        let base = summary_bim_model(levels = [level(0.0)],
                                     fixtures = [family_element(xyz(1.0, 2.0, 0.0))]),
            m = merge(base, (fallback_meshes = [:m1, :m2],)),
            s = model_summary(m)
          @test s["count.fixtures"] == 1
          @test s["count.fallback_meshes"] == 2
          @test s["bbox"] == (1.0, 2.0, 0.0, 1.0, 2.0, 0.0)
        end
      end

      @testset "empty model: no bbox key, empty elevations" begin
        let s = model_summary(summary_bim_model())
          @test !haskey(s, "bbox")
          @test s["level_elevations"] == Float64[]
          @test s["total_wall_length"] == 0.0
          @test all(s["count.$c"] == 0 for c in KhepriBase.summary_count_categories)
        end
      end

      @testset "serialization roundtrip" begin
        let s = model_summary(sample_model()),
            path = joinpath(mktempdir(), "summary.txt")
          write_summary(path, s)
          let s2 = read_summary(path)
            @test summary_string(s2) == summary_string(s)
            @test s2["count.walls"] === 4
            @test s2["level_elevations"] == [0.0, 3.0]
            @test s2["bbox"] == (0.0, 0.0, 0.0, 6.0, 4.0, 0.0)
            @test s2["total_slab_area"] == 24.0
          end
          # keys sorted, one per line, trailing newline
          let str = summary_string(s)
            @test endswith(str, "\n")
            let ks = [String(strip(split(ln, '=')[1])) for ln in split(strip(str), '\n')]
              @test ks == sort(ks)
            end
          end
          # empty model roundtrips too (no bbox line, empty elevations line)
          let e = model_summary(summary_bim_model()),
              epath = joinpath(mktempdir(), "empty.txt")
            write_summary(epath, e)
            let e2 = read_summary(epath)
              @test !haskey(e2, "bbox")
              @test e2["level_elevations"] == Float64[]
              @test summary_string(e2) == summary_string(e)
            end
          end
        end
      end

      @testset "compare_summaries: identical models pass" begin
        let s = model_summary(sample_model()),
            r = compare_summaries(s, s)
          @test r.ok
          @test isempty(fail_lines(r))
        end
      end

      @testset "compare_summaries: 3 extra template levels" begin
        let s = model_summary(sample_model()),
            rb = copy(s)
          rb["count.levels"] = s["count.levels"] + 3
          rb["level_elevations"] = sort(vcat(s["level_elevations"], [0.05, 2.7, 4.2]))
          let r = compare_summaries(s, rb, allow = [:template_levels])
            @test r.ok
            @test !isempty(warn_lines(r))
          end
          # without the allowance the extra levels FAIL
          @test !compare_summaries(s, rb).ok
          # a 4th extra level exceeds the allowance
          rb["count.levels"] += 1
          rb["level_elevations"] = sort(vcat(rb["level_elevations"], [7.9]))
          @test !compare_summaries(s, rb, allow = [:template_levels]).ok
          # missing src levels still FAIL even with the allowance
          let rb2 = copy(s)
            rb2["count.levels"] = 1
            rb2["level_elevations"] = [0.0]
            @test !compare_summaries(s, rb2, allow = [:template_levels]).ok
          end
        end
      end

      @testset "compare_summaries: missing wall fails" begin
        let s = model_summary(sample_model()),
            rb = copy(s)
          rb["count.walls"] = s["count.walls"] - 1
          let r = compare_summaries(s, rb)
            @test !r.ok
            @test any(occursin("count.walls", ln) for ln in fail_lines(r))
          end
        end
      end

      @testset "compare_summaries: stair-generated railings" begin
        let st = stair(u0(), bottom_level = level(0.0), top_level = level(3.0)),
            src = model_summary(summary_bim_model(levels = [level(0.0), level(3.0)],
                                                  stairs = [st])),
            rb = copy(src)
          rb["count.railings"] = src["count.railings"] + 2
          @test compare_summaries(src, rb, allow = [:stair_railings]).ok
          @test !compare_summaries(src, rb).ok
          # +3 exceeds the 2-per-stair allowance for 1 stair
          rb["count.railings"] = src["count.railings"] + 3
          @test !compare_summaries(src, rb, allow = [:stair_railings]).ok
        end
      end

      @testset "compare_summaries: fallback mesh differences always WARN" begin
        let s = model_summary(sample_model()),
            rb = copy(s)
          rb["count.fallback_meshes"] = 5
          let r = compare_summaries(s, rb)
            @test r.ok
            @test any(occursin("count.fallback_meshes", ln) for ln in warn_lines(r))
          end
        end
      end

      @testset "compare_summaries: tolerances" begin
        let s = model_summary(sample_model()),
            rb = copy(s)
          # elevations within elev_atol match
          rb["level_elevations"] = [0.001, 3.001]
          @test compare_summaries(s, rb).ok
          # beyond elev_atol they don't
          rb["level_elevations"] = [0.01, 3.01]
          @test !compare_summaries(s, rb).ok
          # wall length within 1% rtol passes, beyond fails
          let rb2 = copy(s)
            rb2["total_wall_length"] = s["total_wall_length"] * 1.005
            @test compare_summaries(s, rb2).ok
            rb2["total_wall_length"] = s["total_wall_length"] * 1.05
            @test !compare_summaries(s, rb2).ok
          end
          # count_tol turns a small count delta into WARN
          let rb3 = copy(s)
            rb3["count.columns"] = s["count.columns"] - 1
            @test !compare_summaries(s, rb3).ok
            let r = compare_summaries(s, rb3, count_tol = Dict("count.columns" => 1))
              @test r.ok
              @test any(occursin("count.columns", ln) for ln in warn_lines(r))
            end
          end
          # bbox: missing in rebuilt fails
          let rb4 = copy(s)
            delete!(rb4, "bbox")
            @test !compare_summaries(s, rb4).ok
          end
        end
      end

    end
  end
end
