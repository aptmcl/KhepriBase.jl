@testset "Review regression fixes" begin
  @testset "resource path validation" begin
    @test KhepriBase.resource_relative_path("textures/wood.png") == normpath("textures/wood.png")
    @test KhepriBase.resource_relative_path("../Project.toml") === nothing
    @test KhepriBase.resource_relative_path("textures/%2e%2e/Project.toml") === nothing
    @test KhepriBase.resource_relative_path("/absolute/path") === nothing
    @test KhepriBase.resource_relative_path("textures\\..\\Project.toml") === nothing

    root = mktempdir()
    mkpath(joinpath(root, "textures"))
    write(joinpath(root, "textures", "wood.png"), "ok")
    KhepriBase.add_resource_folder!(root)
    @test KhepriBase.http_response_with_resource_file("textures/wood.png").status == 200
    @test KhepriBase.http_response_with_resource_file("../Project.toml").status == 400
  end

  @testset "generic Any protocol shape" begin
    io = IOBuffer()
    KhepriBase.encode(Val(:CS), Val(:Any), io, rgb(0.1, 0.2, 0.3))
    bytes = take!(io)
    @test bytes[1] == 0x07
    @test length(bytes) == 1 + 3*sizeof(Float32)

    io = IOBuffer()
    KhepriBase.encode(Val(:CS), Val(:Any), io, rgba(0.1, 0.2, 0.3, 0.4))
    bytes = take!(io)
    @test bytes[1] == 0x08
    @test length(bytes) == 1 + 4*sizeof(Float32)

    io = IOBuffer()
    KhepriBase.encode(Val(:CS), Val(:Any), io, Dict("outer" => Dict("color" => rgb(1, 0, 0))))
    bytes = take!(io)
    @test bytes[1] == 0x09
    @test 0x07 in bytes
  end

  @testset "path sequence and angle edge cases" begin
    @test KhepriBase.is_empty_path(path_sequence(empty_path(), empty_path()))
    @test_throws ArgumentError angle_between(vx(0), vx(1))
    @test angle_between(vx(1), vx(1)) ≈ 0 atol=1e-12
  end

  @testset "stair step count" begin
    @test KhepriBase.stair_step_count(3.0, 0.18) == Int(round(3.0 / 0.18))
  end

  @testset "fallback frustum and circular curve extrusion" begin
    b = mock_backend()
    reset_mock_backend!(b)
    b_extruded_curve(b, circular_path(u0(), 1), vz(1), u0(), nothing)
    @test length(b.cylinders) == 1
    @test b.cylinders[1].bottom_material === nothing
    @test b.cylinders[1].top_material === nothing
    @test isempty(b.triangles)

    n = KhepriBase.tessellation_divisions(b)
    side_tris = 2n
    cap_tris = n - 2

    reset_mock_backend!(b)
    KhepriBase.b_cylinder_surfaces(b, u0(), 1, 1, nothing, nothing, :side)
    @test isempty(b.cylinders)
    @test length(b.triangles) == side_tris

    reset_mock_backend!(b)
    KhepriBase.b_cylinder_surfaces(b, u0(), 1, 1, nothing, :top, :side)
    @test isempty(b.cylinders)
    @test length(b.triangles) == side_tris + cap_tris

    reset_mock_backend!(b)
    KhepriBase.b_cylinder_surfaces(b, u0(), 1, 1, :bottom, :top, :side)
    @test isempty(b.cylinders)
    @test length(b.triangles) == side_tris + 2cap_tris

    reset_mock_backend!(b)
    bottom = closed_polygonal_path([xy(0, 0), xy(1, 0), xy(1, 1), xy(0, 1)])
    top = closed_polygonal_path([xy(0, 0), xy(1, 0), xy(1.5, 0.5), xy(1, 1), xy(0, 1), xy(-0.5, 0.5)])
    KhepriBase.b_path_frustum(b, bottom, top, nothing, nothing, nothing)
    @test length(b.triangles) >= 2 * length(path_vertices(top))
  end

  @testset "surface CSG fallback fails clearly" begin
    c0 = surface_circle(xy(0, 0), 1)
    c1 = surface_circle(xy(0.5, 0), 1)
    @test_throws ArgumentError KhepriBase.shape_path(union(c0, c1))
  end
end
