#=
Architectural material library — single source of truth.

Why this file exists. Khepri's promise is "same script, different backend".
For geometry the promise is straightforward; for appearance it only holds
if every backend maps each canonical material (material_metal,
material_glass, …) to visually comparable output. Without a shared
definition, every backend package grows its own lookup table (see the
BlenderKit-asset-id table that used to live in KhepriBlender), those tables
drift, and a metal sphere looks chrome in one backend and matte white in
another. This module is the one place those values live.

The canonical PbrMaterial constants themselves live in Shapes.jl (so that
`using KhepriBase; material_metal` just works); this file provides an
iterable view of them plus a small accessor for backend integrations that
want the parameters as a plain NamedTuple.

Who uses this.
  - Cross-backend material regression test
    (KhepriBase/test/test_arch_materials.jl).
  - User-facing material catalogue (Docs/Materials.md).
  - Any backend package that wants to pre-register hand-tuned native
    materials rather than going through the generic `b_material` cascade —
    e.g., KhepriAutoCAD with its RapidRT material presets.

Backends that *don't* need custom mappings get correct behaviour for free
via `realize(backend, PbrMaterial)` → `b_material(backend, ...)` cascading
through the PBR tiers defined in Backend.jl.
=#

export architectural_materials, architectural_material_spec

"""Tuple of every canonical architectural material, in documentation order."""
const architectural_materials = (
  material_basic,
  material_metal,
  material_glass,
  material_wood,
  material_concrete,
  material_plaster,
  material_grass,
  material_clay,
)

"""
    architectural_material_spec(mat::PbrMaterial) → NamedTuple

Return the full PBR parameter set of `mat` as a keyword-style NamedTuple
suitable for driving any backend's native material constructor. Fields
match the parameters of `b_material`.
"""
architectural_material_spec(mat::PbrMaterial) = (
  name         = mat.name,
  base_color   = mat.base_color,
  metallic     = mat.metallic,
  roughness    = mat.roughness,
  reflectance  = mat.reflectance,
  ior          = mat.ior,
  transmission = mat.transmission,
  transmission_roughness = mat.transmission_roughness,
  clearcoat             = mat.clear_coat,
  clearcoat_roughness   = mat.clear_coat_roughness,
  emission_color        = mat.emissive,
  emission_strength     = 1.0,
)

"""
    architectural_material_spec(name::Symbol) → NamedTuple

Look up a canonical architectural material by short name:
`:basic, :metal, :glass, :wood, :concrete, :plaster, :grass, :clay`.
"""
function architectural_material_spec(name::Symbol)
  target = Symbol("material_$(name)")
  for m in architectural_materials
    if Symbol("material_$(m.name)") == Symbol("material_$(lowercase(m.name))") &&
       Symbol(lowercase(m.name)) == name
      return architectural_material_spec(m)
    end
  end
  error("Unknown architectural material $name. Valid: :basic, :metal, :glass, :wood, :concrete, :plaster, :grass, :clay")
end
