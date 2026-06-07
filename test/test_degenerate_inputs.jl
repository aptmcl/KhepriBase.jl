# test_degenerate_inputs.jl — rank-8 regression tests.
#
# A family of path/shape/space ops indexed [1]/[end] or reduced without init on
# empty/single-element collections, throwing opaque BoundsError/ArgumentError
# where the codebase elsewhere guards (bounding_box, _space_bbox). These pin the
# defensive guards so the degenerate cases stay actionable, not crashes.
using Test
using KhepriBase

@testset "Degenerate inputs (rank 8)" begin
  @testset "is_closed_path / polygonal_path / path_length on empty/single (PATHS-1)" begin
    # is_closed_path(::Locs): empty and single-point are NOT closed (no BoundsError)
    @test is_closed_path(KhepriBase.Loc[]) == false
    @test is_closed_path([u0()]) == false
    @test is_closed_path([u0(), x(1), u0()]) == true   # genuine closed loop still detected

    # polygonal_path: fewer than two vertices throws a descriptive ArgumentError
    @test_throws ArgumentError polygonal_path(KhepriBase.Loc[])
    @test_throws ArgumentError polygonal_path([u0()])
    @test_throws ArgumentError polygonal_path(u0())     # single-vertex splat form
    @test polygonal_path([u0(), x(1)]) isa KhepriBase.OpenPolygonalPath

    # path_length(::Locs): empty list measures 0.0 (no BoundsError)
    @test path_length(KhepriBase.Loc[]) == 0.0
    @test path_length([u0()]) == 0.0
    @test path_length([u0(), x(3)]) ≈ 3.0
  end

  @testset "join_paths on a single-vertex path (PATHS-3)" begin
    # A 1-vertex path used to make the collinearity test index vertices[end-1]
    # == vertices[0] (or p2.vertices[2]) -> BoundsError; now falls back to
    # plain concatenation.
    p1short = open_polygonal_path([xy(1, 0)])               # 1 vertex
    p2 = open_polygonal_path([xy(1, 0), xy(1, 1)])          # coincident start
    @test path_vertices(join_paths(p1short, p2)) == [xy(1, 0), xy(1, 1)]

    p1 = open_polygonal_path([xy(0, 0), xy(1, 0)])
    p2short = open_polygonal_path([xy(1, 0)])               # 1 vertex, coincident
    @test path_vertices(join_paths(p1, p2short)) == [xy(0, 0), xy(1, 0)]
  end

  @testset "convert(ClosedPath, ::PathSet) on empty/singleton (PATHS-4)" begin
    # Empty PathSet has no outer boundary -> descriptive error, not the opaque
    # "reduce over empty collection".
    @test_throws ArgumentError convert(ClosedPath, KhepriBase.PathSet(KhepriBase.Path[]))
    # Singleton PathSet holding an OPEN path must still yield a ClosedPath.
    open_p = open_polygonal_path([xy(0, 0), xy(1, 0), xy(1, 1)])
    @test convert(ClosedPath, path_set(open_p)) isa ClosedPath
    # Multi-path case still folds via subtract_paths into a ClosedPath.
    outer = closed_polygonal_path([xy(0, 0), xy(4, 0), xy(4, 4), xy(0, 4)])
    inner = closed_polygonal_path([xy(1, 1), xy(2, 1), xy(2, 2), xy(1, 2)])
    @test convert(ClosedPath, path_set(outer, inner)) isa ClosedPath
  end

  @testset "degenerate composite -> polygonal conversion (PATHS-16)" begin
    # A CompositePath whose pieces contribute no vertices (a 1-vertex piece is
    # emptied by the [1:end-1] slice; a 0-vertex piece adds nothing) leaves the
    # accumulator empty; the convert must refuse with a clear ArgumentError
    # rather than a BoundsError deep in closed_polygonal_path([]).
    degc = KhepriBase.CompositePath{true}(KhepriBase.OpenPath[open_polygonal_path([xy(1, 1)])])
    @test_throws ArgumentError convert(KhepriBase.ClosedPolygonalPath, degc)
    dego = KhepriBase.CompositePath{false}(KhepriBase.OpenPath[open_polygonal_path(KhepriBase.Loc[])])
    @test_throws ArgumentError convert(KhepriBase.OpenPolygonalPath, dego)
  end

  @testset "composite conversion is type-stable (PERF-11)" begin
    p1 = open_polygonal_path([xy(0, 0), xy(1, 0)])
    p2 = open_polygonal_path([xy(1, 0), xy(1, 1)])
    p3 = open_polygonal_path([xy(1, 1), xy(0, 0)])
    seq = closed_path_sequence(p1, p2, p3)
    @test convert(KhepriBase.ClosedPolygonalPath, seq).vertices isa KhepriBase.Locs
  end

  @testset "loft on empty/single profile list (SHAPES-3)" begin
    # A loft needs >= 2 cross sections. Empty input must not vacuously route
    # through all(is_point, []) into a point loft of nothing; a single profile
    # is geometrically meaningless. Both must error.
    @test_throws ErrorException loft(KhepriBase.Shape[])
    @test_throws ErrorException loft([point(u0())])
  end

  @testset "bounding_rectangle on no points (SHAPES-4)" begin
    @test_throws ErrorException KhepriBase.bounding_rectangle(KhepriBase.Loc[])
    # non-empty still works (compare coordinates; XYZ `==` is not value-based)
    br = KhepriBase.bounding_rectangle([xyz(0, 0, 0), xyz(2, 3, 0)])
    @test length(br) == 2
    @test (cx(br[1]), cy(br[1])) == (0.0, 0.0)
    @test (cx(br[2]), cy(br[2])) == (2.0, 3.0)
  end

  @testset "polygon_area / space_perimeter on empty boundary (SPACES-10)" begin
    # An empty ClosedPolygonalPath bypasses ensure_no_repeated_locations via the
    # inner struct constructor, reaching the measures with zero vertices
    # (regression: mod1(i+1,0) / empty-sum errors).
    @test KhepriBase.polygon_area(KhepriBase.Loc[]) == 0.0
    empty_space = KhepriBase.Space(:empty, :space, KhepriBase.ClosedPolygonalPath(KhepriBase.Loc[]))
    @test space_area(empty_space) == 0.0
    @test space_perimeter(empty_space) == 0.0
    # Non-empty path still computes correctly (unit square).
    sq_space = KhepriBase.Space(:sq, :space,
                                KhepriBase.ClosedPolygonalPath([u0(), x(1), xy(1, 1), y(1)]))
    @test space_area(sq_space) ≈ 1.0
    @test space_perimeter(sq_space) ≈ 4.0
  end
end
