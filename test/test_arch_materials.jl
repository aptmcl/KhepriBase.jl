# test_arch_materials.jl — architectural material library contract tests.
#
# These tests run without any backend loaded; they assert the shape of the
# canonical material library and the accessor that backends consume. Full
# cross-backend rendered-comparison lives in
# `Julia/KhepriMitsuba/test/live_arch_materials.jl` and
# `Julia/KhepriBlender/test/live_arch_materials.jl`, which need live
# Mitsuba / Blender connections.

using KhepriBase
using Test

@testset "architectural_materials tuple" begin
  @test length(architectural_materials) == 8
  names = [m.name for m in architectural_materials]
  @test names == ["Basic", "Metal", "Glass", "Wood", "Concrete",
                  "Plaster", "Grass", "Clay"]
  for m in architectural_materials
    @test m isa KhepriBase.PbrMaterial
  end
end

@testset "architectural_material_spec fields" begin
  spec = architectural_material_spec(material_metal)
  # Every field of the NamedTuple must be populated — backends use them all.
  @test spec.name == "Metal"
  @test spec.metallic == 1.0
  @test spec.roughness == 0.15
  @test spec.ior == 1.5
  @test spec.transmission == 0.0
  @test spec.base_color isa KhepriBase.RGBA

  gspec = architectural_material_spec(material_glass)
  @test gspec.transmission == 0.95
  @test gspec.roughness == 0.0
  @test gspec.ior == 1.5

  # Every canonical material yields a spec with non-nothing fields.
  for m in architectural_materials
    s = architectural_material_spec(m)
    @test s.name isa String
    @test s.base_color isa KhepriBase.RGBA
    @test 0.0 <= s.metallic <= 1.0
    @test 0.0 <= s.roughness <= 1.0
    @test s.ior >= 1.0
    @test 0.0 <= s.transmission <= 1.0
  end
end

@testset "architectural library invariants" begin
  # The exactly-one-metal invariant: only material_metal should have
  # metallic=1.0. Everyone else is a dielectric.
  metals = filter(m -> m.metallic == 1.0, collect(architectural_materials))
  @test length(metals) == 1
  @test metals[1] === material_metal

  # Only material_glass is transmissive; tune thresholds if we add more
  # transparent materials (frosted plastic, sheer curtains, etc.).
  transmissive = filter(m -> m.transmission > 0.01, collect(architectural_materials))
  @test length(transmissive) == 1
  @test transmissive[1] === material_glass

  # Roughness ordering: grass > concrete, clay > plaster > wood > metal
  # (standard architectural surface-finish intuition).
  @test material_grass.roughness >= material_concrete.roughness
  @test material_concrete.roughness >= material_plaster.roughness
  @test material_plaster.roughness >= material_wood.roughness
  @test material_wood.roughness >= material_metal.roughness
end

@testset "exports" begin
  # User-facing symbols are exported from KhepriBase.
  @test isdefined(KhepriBase, :architectural_materials)
  @test isdefined(KhepriBase, :architectural_material_spec)
  for sym in (:material_basic, :material_metal, :material_glass, :material_wood,
              :material_concrete, :material_plaster, :material_grass, :material_clay)
    @test isdefined(KhepriBase, sym)
  end
end
