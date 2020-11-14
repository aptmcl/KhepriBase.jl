#=
A Region is a (possibly infinite) set of locations in space.
Regions include points, curves, surfaces, and volumes
A ParametricRegion and an ImplicitRegion are examples of a Region.
=#

abstract type Region end
show(io::IO, s::Region) =
  print(io, "$(typeof(s))(...)")

Regions = Vector{<:Region}

abstract type Region0D <: Region end
abstract type Region1D <: Region end
abstract type Region2D <: Region end
abstract type Region3D <: Region end

macro defregion(name_typename, parent, fields...)
  (name, typename) = name_typename isa Symbol ?
    (name_typename, Symbol(string(map(uppercasefirst,split(string(name_typename),'_'))...))) :
    name_typename.args
  name_str = string(name)
  struct_name = esc(typename)
  field_names = map(field -> field.args[1].args[1], fields)
  field_types = map(field -> esc(field.args[1].args[2]), fields)
  field_inits = map(field -> field.args[2], fields)
  field_renames = map(Symbol ∘ string, field_names)
  field_replacements = Dict(zip(field_names, field_renames))
  struct_fields = map((name,typ) -> :($(name) :: $(typ)), field_names, field_types)
  mk_param(name,typ,init) = Expr(:kw, name, init)
  opt_params = map(mk_param, field_renames, field_types, map(init -> replace_in(init, field_replacements), field_inits))
  key_params = map(mk_param, field_names, field_types, field_renames)
  constructor_name = esc(name)
  predicate_name = esc(Symbol("is_", name_str))
  mk_convert(name,typ) = :($(esc(name)))
  field_converts = map(mk_convert, field_names, field_types)
  selector_names = map(field_name -> esc(Symbol(name_str, "_", string(field_name))), field_names)
  quote
    export $(constructor_name), $(struct_name), $(predicate_name) #, $(selector_names...)
    struct $struct_name <: $parent
      $(struct_fields...)
    end
    @noinline $(constructor_name)($(opt_params...); $(key_params...)) =
      $(struct_name)($(field_converts...))
    $(predicate_name)(v::$(struct_name)) = true
    $(predicate_name)(v::Any) = false
    #$(map((selector_name, field_name) -> :($(selector_name)(v::$(struct_name)) = v.$(field_name)),
    #      selector_names, field_names)...)
    KhepriBase.meta_program(v::$(struct_name)) =
        Expr(:call, $(Expr(:quote, name)), $(map(field_name -> :(meta_program(v.$(field_name))), field_names)...))
  end
end

@defregion(annulus, Region2D, center::Loc=u0(), outer_radius::Real=1, inner_radius::Real=1/2)

area(r::Annulus) = annulus_area(r.outer_radius, r.inner_radius)
