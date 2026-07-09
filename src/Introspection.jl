# Introspection.jl — portable data types for reading a CAD/BIM model back into Khepri.
#
# This file hosts the backend-agnostic vocabulary shared by the introspection (backend model →
# Khepri shapes) and code-generation (shapes → Julia program) halves of the BIM→ABIM pipeline.
# The transform/emit passes live in CodeGen.jl; the per-backend element queries live in each
# backend. For now this defines the family-metadata carrier; the abstract `b_all_*` read API is
# added alongside a generic `introspect_model` in a later phase.

export FamilyMeta, family_key

# Metadata captured for a reconstructed family during introspection. Threaded through the model
# (see `model.family_meta`) and consumed by the code-generation passes to name family variables and
# emit their backend mappings. Replaces the former global `_introspection_meta`/`_family_expr_meta`
# dicts. `category` is the Khepri family category (:wall, :door, :column, …); `path` is the loadable
# family file (e.g. a Revit `.rfa`), empty for system families; `obj_name` is the name of the OBJ/MTL
# mesh extracted for this family (empty if none), used to emit a cross-backend mapping so the same
# generated call reproduces as the real mesh on non-BIM backends (KhepriThreejs).
struct FamilyMeta
  category::Symbol
  family_name::String
  type_name::String
  is_system::Bool
  path::String
  obj_name::String
end

FamilyMeta(; category, family_name, type_name, is_system, path="", obj_name="") =
  FamilyMeta(category, family_name, type_name, is_system, path, obj_name)

# Stable identity of a family type; used both to name the family at reconstruction (so that
# distinct native types produce distinct `meta_program` Exprs — avoiding the type-collapse where all
# doors emitted the identical `door_family()`) and to key/deduplicate metadata.
family_key(m::FamilyMeta) = "$(m.family_name):$(m.type_name)"
