export GenericRef,
       EmptyRef,
       UniversalRef,
       NativeRef,
       UnionRef,
       SubtractionRef,
       DynRef,
       DynRefs,
       void_ref,
       ensure_ref,
       map_ref,
       collect_ref,
       unite_ref,
       intersect_ref,
       subtract_ref
export Shape,
       Shapes,
       Backend,
       LazyBackend,
       Path,
       backend,
       new_backend,
       backend_name,
       current_backend,
       has_current_backend,
       switch_to_backend,
       delete_shape, delete_shapes,
       delete_all_shapes, mark_deleted,
       force_realize,
       set_length_unit,
       is_collecting_shapes,
       collecting_shapes,
       collected_shapes,
       with_transaction,
       surface_boundary,
       curve_domain,
       surface_domain,
       get_layer,
       create_layer,
       get_or_create_layer,
       current_layer,
       delete_all_shapes_in_layer,
       get_material,
       create_material,
       get_or_create_material,
       create_block,
       instantiate_block,
       reset_backend,
       connection,
       @deffamily,
       @defproxy,
       force_creation,
       subpath,
       subpath_starting_at,
       subpath_ending_at,
       bounding_box,
       capture_shape, capture_shapes,
       captured_shape, captured_shapes,
       revolve

#References can be (single or multiple) native references
abstract type GenericRef{K,T} end

struct EmptyRef{K,T} <: GenericRef{K,T} end
struct UniversalRef{K,T} <: GenericRef{K,T} end

struct NativeRef{K,T} <: GenericRef{K,T}
  value::T
end
struct NativeRefs{K,T} <: GenericRef{K,T}
  values::Vector{T}
end

ensure_ref(b::Backend{K,T}, r::T) where {K,T} = NativeRef{K,T}(r)
ensure_ref(b::Backend{K,T}, rs::Vector{T}) where {K,T} =
  length(rs) == 1 ?
    NativeRef{K,T}(rs[1]) :
    NativeRefs{K,T}(rs)

#Unions and subtractions are needed because actual backends frequently fail those operations
struct UnionRef{K,T} <: GenericRef{K,T}
  values::Tuple{Vararg{GenericRef{K,T}}}
end
struct SubtractionRef{K,T} <: GenericRef{K,T}
  value::GenericRef{K,T}
  values::Tuple{Vararg{GenericRef{K,T}}}
end

ensure_ref(b::Backend{K,T}, v::GenericRef{K,T}) where {K,T} = v
ensure_ref(b::Backend{K,T}, v::Vector{<:S}) where {K,T,S} =
  length(v) == 1 ?
    ensure_ref(b, v[1]) :
    UnionRef{K,T}(Tuple((ensure_ref(b, vi) for vi in v)))

# currying
map_ref(b::Backend{K,T}, f::Function) where {K,T} = r -> map_ref(b, f, r)

map_ref(b::Backend{K,T}, f::Function, r::NativeRef{K,T}) where {K,T} = ensure_ref(b, f(r.value))
map_ref(b::Backend{K,T}, f::Function, r::UnionRef{K,T}) where {K,T} = UnionRef{K,T}(map(map_ref(b, f), r.values))
map_ref(b::Backend{K,T}, f::Function, r::SubtractionRef{K,T}) where {K,T} = SubtractionRef{K,T}(map_ref(b, f, r.value), map(map_ref(b, f), r.values))
map_ref(b::Backend{K,T}, f::Function, r::NativeRefs{K,T}) where {K,T} = ensure_ref(b, map(r -> map_ref(b, f, NativeRef{K,T}(r)), r.values))


# currying
collect_ref(b::Backend{K,T}) where {K,T} = r -> collect_ref(b, r)

collect_ref(b::Backend{K,T}, r::EmptyRef{K,T}) where {K,T} = T[]
collect_ref(b::Backend{K,T}, r::NativeRef{K,T}) where {K,T} = T[r.value]
collect_ref(b::Backend{K,T}, r::UnionRef{K,T}) where {K,T} =
  mapreduce(collect_ref(b), vcat, r.values, init=T[])
collect_ref(b::Backend{K,T}, r::SubtractionRef{K,T}) where {K,T} =
  vcat(collect_ref(b, r.value), mapreduce(collect_ref(b), vcat, r.values, init=T[]))

# Boolean algebra laws
# currying
unite_ref(b::Backend{K,T}) where {K,T} = (r0::GenericRef{K,T}, r1::GenericRef{K,T}) -> unite_ref(b, r0, r1)

unite_ref(b::Backend{K,T}, r0::GenericRef{K,T}, r1::UniversalRef{K,T}) where {K,T} = r1
unite_ref(b::Backend{K,T}, r0::UniversalRef{K,T}, r1::GenericRef{K,T}) where {K,T} = r0

#To avoid ambiguity
unite_ref(b::Backend{K,T}, r0::UnionRef{K,T}, r1::UnionRef{K,T}) where {K,T} =
  unite_ref(b, unite_refs(b, r0), unite_refs(b, r1))
unite_ref(b::Backend{K,T}, r0::EmptyRef{K,T}, r1::EmptyRef{K,T}) where {K,T} = r0
unite_ref(b::Backend{K,T}, r0::UnionRef{K,T}, r1::EmptyRef{K,T}) where {K,T} = r0
unite_ref(b::Backend{K,T}, r0::EmptyRef{K,T}, r1::UnionRef{K,T}) where {K,T} = r1
unite_ref(b::Backend{K,T}, r0::GenericRef{K,T}, r1::EmptyRef{K,T}) where {K,T} = r0
unite_ref(b::Backend{K,T}, r0::EmptyRef{K,T}, r1::GenericRef{K,T}) where {K,T} = r1

unite_refs(b::Backend{K,T}, r::UnionRef{K,T}) where {K,T} =
  foldr((r0,r1)->unite_ref(b,r0,r1), r.values, init=EmptyRef{K,T}())
unite_ref(b::Backend{K,T}, r0::UnionRef{K,T}, r1::GenericRef{K,T}) where {K,T} =
  unite_ref(b, unite_refs(b, r0), r1)
unite_ref(b::Backend{K,T}, r0::GenericRef{K,T}, r1::UnionRef{K,T}) where {K,T} =
  unite_ref(b, r0, unite_refs(b, r1))

# currying
intersect_ref(b::Backend{K,T}) where {K,T} = (r0::GenericRef{K,T}, r1::GenericRef{K,T}) -> intersect_ref(b, r0, r1)

intersect_ref(b::Backend{K,T}, r0::GenericRef{K,T}, r1::UniversalRef{K,T}) where {K,T} = r0
intersect_ref(b::Backend{K,T}, r0::UniversalRef{K,T}, r1::GenericRef{K,T}) where {K,T} = r1
intersect_ref(b::Backend{K,T}, r0::GenericRef{K,T}, r1::EmptyRef{K,T}) where {K,T} = r1
intersect_ref(b::Backend{K,T}, r0::EmptyRef{K,T}, r1::GenericRef{K,T}) where {K,T} = r0
intersect_ref(b::Backend{K,T}, r0::GenericRef{K,T}, r1::UnionRef{K,T}) where {K,T} =
  intersect_ref(b, r0, unite_refs(b, r1))
intersect_ref(b::Backend{K,T}, r0::UnionRef{K,T}, r1::GenericRef{K,T}) where {K,T} =
  intersect_ref(b, unite_refs(b, r0), r1)

#To avoid ambiguity
# currying
subtract_ref(b::Backend{K,T}) where {K,T} = (r0::GenericRef{K,T}, r1::GenericRef{K,T}) -> subtract_ref(b, r0, r1)

subtract_ref(b::Backend{K,T}, r0::UnionRef{K,T}, r1::UnionRef{K,T}) where {K,T} =
  subtract_ref(b, unite_refs(b, r0), unite_refs(b, r1))
subtract_ref(b::Backend{K,T}, r0::GenericRef{K,T}, r1::UniversalRef{K,T}) where {K,T} = EmptyRef{K,T}()
subtract_ref(b::Backend{K,T}, r0::GenericRef{K,T}, r1::EmptyRef{K,T}) where {K,T} = r0
subtract_ref(b::Backend{K,T}, r0::EmptyRef{K,T}, r1::GenericRef{K,T}) where {K,T} = r0
subtract_ref(b::Backend{K,T}, r0::GenericRef{K,T}, r1::UnionRef{K,T}) where {K,T} =
  subtract_ref(b, r0, unite_refs(b, r1))
subtract_ref(b::Backend{K,T}, r0::UnionRef{K,T}, r1::GenericRef{K,T}) where {K,T} =
  subtract_ref(b, unite_refs(b, r0), r1)

# References need to be created, deleted, and recreated, depending on the way the backend works
# For example, each time a shape is consumed, it becomes deleted and might need to be recreated
mutable struct DynRef{K,R}
  backend::Backend{K,R}
  value::GenericRef{K,R}
  created::Int
  deleted::Int
end

#DynRef{K,R}(backend::Backend{K,T}) where {K,R} = DynRef{K,R}(backend, void_ref(backend), 0, 0)
DynRef(b::Backend{K,T}, v) where {K,T} = DynRef{K,T}(b, ensure_ref(b, v), 1, 0)


const DynRefs = IdDict{Backend, Any}
const dyn_refs = DynRefs

abstract type Proxy end

backend(s::Proxy) = first(first(s.ref))

realized_in(s::Proxy, b::Backend) = s.ref[b].created == s.ref[b].deleted + 1
# This is so stupid. We need call-next-method.
really_mark_deleted(b::Backend, s::Proxy) = really_mark_deleted(b, s.ref)
really_mark_deleted(b::Backend, ref::DynRefs) = delete!(ref, b)
really_mark_deleted(b::Backend, s::Any) = nothing
mark_deleted(b::Backend, s::Proxy) = really_mark_deleted(b, s)
# We also need to propagate this to all dependencies
mark_deleted(b::Backend, ss::Array{<:Proxy}) = foreach(s->mark_deleted(b, s), ss)
mark_deleted(b::Backend, s::Any) = nothing
marked_deleted(b::Backend, s::Proxy) = !haskey(s.ref, b)

#=
The protocol is this:
ref(b, s) calls
  force_realize(b, s)
=#

ref(b::Backend, s::Proxy) =
  force_realize(b, s)
ref(b::Backend) =
  s::Proxy -> ref(b, s)

reset_ref(b::Backend, s::Proxy) =
  delete!(s.ref, b)

force_realize(b::Backend, s::Proxy) =
  haskey(s.ref, b) ?
    s.ref[b] : #error("Shape was already realized in $(b)") :
    s.ref[b] = ensure_ref(b, realize(b, s))

realized(b::Backend, s::Proxy) =
  haskey(s.ref, b)

force_realize(s::Proxy) =
  for b in current_backends()
    force_realize(b, s)
  end

# We can also use a shape as a surrogate for another shape

ensure_ref(b::Backend{K,T}, v::Proxy) where {K,T} =
  ref(b, v)

abstract type Shape <: Proxy end
show(io::IO, s::Shape) =
  print(io, "$(typeof(s))(...)")

Shapes = Vector{<:Shape}

map_ref(f::Function, b::Backend, s::Shape) = map_ref(b, f, ref(b, s))
collect_ref(s::Shape) = error("collect_ref(s.ref.backend, ref(s))")
collect_ref(ss::Shapes) = error("mapreduce(collect_ref, vcat, ss, init=[])")

#=
Whenever a shape is created, it might be eagerly realized in its backend,
depending on the kind of shape and kind of backend (and/or its current state).
Another possibility is for the shape to be saved in some container.
It might also be necessary to record the control flow that caused the shape to be created.
This means that we need to control what happens immediately after a shape is initialized.
The protocol after_init takes care of that.
=#

after_init(a::Any) = a
after_init(s::Shape) =
  begin
    maybe_realize(s)
    maybe_collect(s)
    maybe_trace(s)
    s
  end

#=
Backends might need to immediately realize a shape while supporting further modifications
e.g., using boolean operations. Others, however, cannot do that and can only realize
shapes by request, presumably, when they have complete information about them.
A middle term might be a backend that supports both modes.
=#
delay_realize(b::Backend, s::Shape) =
  nothing

delaying_realize = Parameter(false)
with_transaction(fn) =
  maybe_realize(with(fn, delaying_realize, true))

maybe_realize(s::Shape) =
  delaying_realize() ?
    for b in current_backends()
      delay_realize(b, s)
    end :
    for b in current_backends()
      maybe_realize(b, s)
    end

#=
Even if a backend is eager, it might be necessary to temporarily delay the
realization of shapes, particularly, when the construction is incremental.
=#

maybe_realize(b::Backend, s::Shape) =
  if ! realized(b, s)
    force_realize(b, s)
  end

abstract type LazyBackend{K,T} <: Backend{K,T} end
maybe_realize(b::LazyBackend, s::Shape) = delay_realize(b, s)
delay_realize(b::LazyBackend, s::Shape) = save_shape!(b, s)

# By default, save_shape! assumes there is a field in the backend to store shapes
export save_shape!
save_shape!(b::Backend, s::Shape) = (push!(b.shapes, s); s)

#=
Frequently, we need to collect all shapes that are created:
=#

# HACK: Replace in_shape_collection with is_collecting_shapes
in_shape_collection = Parameter(false)
is_collecting_shapes = in_shape_collection
collected_shapes = Parameter(Shape[])
collect_shape!(s::Shape) = (push!(collected_shapes(), s); s)
collecting_shapes(fn) =
    with(collected_shapes, Shape[]) do
        with(in_shape_collection, true) do
            fn()
        end
        collected_shapes()
    end
maybe_collect(s::Shape) = (in_shape_collection() && collect_shape!(s); s)

######################################################
#Traceability
traceability = Parameter(false)
trace_depth = Parameter(1000)
excluded_modules = Parameter([Base, Base.CoreLogging, KhepriBase])
# We a dict from shapes to file locations
# and a dict from file locations to shapes
shape_to_file_locations = IdDict()
file_location_to_shapes = Dict()

export traceability, trace_depth, excluded_modules, clear_trace!, shape_source, source_shapes

shape_source(s) = get(shape_to_file_locations, s, [])
source_shapes(file, line) = get(file_location_to_shapes, (file, line), [])

clear_trace!() =
  begin
    empty!(shape_to_file_locations)
    empty!(file_location_to_shapes)
  end
#=
We do not care about frames that are unrelated to the application.
=#
interesting_locations(frames) =
  let locations = [],
      max_depth = min(trace_depth(), length(frames)-0)#14)
    for i in 2:max_depth
      let frame = frames[i],
          linfo = frame.linfo
        if linfo isa Core.CodeInfo ||
           (linfo isa Core.MethodInstance &&
            ! (linfo.def.module in excluded_modules()))
          push!(locations, (frame.file, frame.line))
        end
      end
    end
    locations
  end

trace!(s) =
  let frames = stacktrace(),
      locations = interesting_locations(frames)
    shape_to_file_locations[s] = locations
    for location in locations
      file_location_to_shapes[location] = Shape[get(file_location_to_shapes, location, [])..., s]
    end
    s
  end

maybe_trace(s) = traceability() && trace!(s)

######################################################

macro defshapeop(name_params)
    name, params = name_params.args[1], name_params.args[2:end]
    quote
        export $(esc(name))
        $(esc(name))(s::Shape, $(map(esc,params)...), b::Backend=backend(s)) =
            throw(UndefinedBackendException())
    end
end

export all_shapes, delete_all_shapes
@defcbs delete_all_shapes()
b_delete_all_shapes(b::Backend) = b_delete_all_refs(b)
@defcb all_shapes()
b_all_shapes(b::Backend) =
  Shape[b_shape_from_ref(b, r) for r in b_all_refs(b)]
@bdef(b_shape_from_ref(r))



@defcbs set_length_unit(unit::String="")
@defcb reset_backend()
@defcb save_as(pathname::String, format::String)

new_backend(b::Backend = top_backend()) = backend(b)

struct WrongTypeForParam <: Exception
  param::Symbol
  value::Any
  expected_type::Type
end
Base.showerror(io::IO, e::WrongTypeForParam) =
  print(io, "$(e.param) expected a $(e.expected_type) but got $(e.value) of type $(typeof(e.value))")

macro defproxy(name_typename, parent, fields...)
  (name, typename) = name_typename isa Symbol ?
    (name_typename, Symbol(string(map(uppercasefirst,split(string(name_typename),'_'))...))) :
    name_typename.args
  name_str = string(name)
  struct_name = esc(typename)
  field_names = map(field -> field.args[1].args[1], fields)
  field_types = map(field -> esc(field.args[1].args[2]), fields)
  field_inits = map(field -> field.args[2], fields)
#  field_renames = map(esc ∘ Symbol ∘ uppercasefirst ∘ string, field_names)
  field_renames = map(Symbol ∘ string, field_names)
  field_replacements = Dict(zip(field_names, field_renames))
  struct_fields = map((name,typ) -> :($(name) :: $(typ)), field_names, field_types)
#  opt_params = map((name,typ,init) -> :($(name) :: $(typ) = $(init)), field_renames, field_types, field_inits)
#  key_params = map((name,typ,rename) -> :($(name) :: $(typ) = $(rename)), field_names, field_types, field_renames)
#  mk_param(name,typ) = Expr(:kw, Expr(:(::), name, typ))
  mk_param(name,typ,init) = Expr(:kw, name, init) #Expr(:kw, Expr(:(::), name, typ), init)
  opt_params = map(mk_param, field_renames, field_types, map(init -> replace_in(init, field_replacements), field_inits))
  key_params = map(mk_param, field_names, field_types, field_renames)
  constructor_name = esc(name)
  predicate_name = esc(Symbol("is_", name_str))
  #mk_convert(name,typ) = :(isa($(esc(name)), $(typ)) ? $(esc(name)) : throw(WrongTypeForParam($(QuoteNode(name)), $(esc(name)), $(typ))))
  mk_convert(name,typ) = :($(esc(name)))
  field_converts = map(mk_convert, field_names, field_types)
  selector_names = map(field_name -> esc(Symbol(name_str, "_", string(field_name))), field_names)
  quote
    export $(constructor_name), $(struct_name), $(predicate_name), $(selector_names...)
    struct $struct_name <: $parent
      ref::DynRefs
      $(struct_fields...)
    end
    # we don't need to convert anything because Julia already does that with the default constructor
    # and, by the same idea, we don't need to define parameter types.
    @noinline $(constructor_name)($(opt_params...); $(key_params...), ref::DynRefs=dyn_refs()) =
      after_init($(struct_name)(ref, $(field_converts...)))
    $(predicate_name)(v::$(struct_name)) = true
    $(predicate_name)(v::Any) = false
    $(map((selector_name, field_name) -> :($(selector_name)(v::$(struct_name)) = v.$(field_name)),
          selector_names, field_names)...)
    KhepriBase.mark_deleted(b::Backend, v::$(struct_name)) =
      if ! marked_deleted(b, v)
        really_mark_deleted(b, v)
        $(map(field_name -> :(mark_deleted(b, v.$(field_name))), field_names)...)
      end
    KhepriBase.meta_program(v::$(struct_name)) =
        Expr(:call, $(Expr(:quote, name)), $(map(field_name -> :(meta_program(v.$(field_name))), field_names)...))
  end
end

#=
There are entities who have parameters that depend on the backend.
We will assume that these entities have one field called data which
should be a BackendParameter.
To assign such a parameter, we use the set_on! function.
=#

export set_on!
set_on!(b::Backend, proxy, ref) =
  begin
    proxy.data(b, ref)
    reset_ref(b, proxy)
    proxy
  end
set_on!(proxy, ref) =
  begin
    proxy.data(ref)
    proxy
  end

#=
Layers are just a classification mechanism.
Some backends, however, can colorize the shapes that have that layer, or can make
those shapes appear and disappear by activating or deactivating the layer.
=#

@defproxy(layer, Proxy, name::String="Layer", active::Bool=true, color::RGB=rgb(1,1,1))
create_layer(args...) =
  let s = layer(args...)
    force_realize(s)
    s
  end
realize(b::Backend, l::Layer) =
  b_layer(b, l.name, l.active, l.color)
current_layer(backends::Backends=current_backends()) =
  layer(ref=DynRefs(b=>ensure_ref(b, b_current_layer(b)) for b in backends))

current_layer(layer, backends::Backends=current_backends()) =
  for b in backends
    b_current_layer(b, ref(b, layer).value)
  end
delete_all_shapes_in_layer(layer, backends::Backends=current_backends()) =
  for b in backends
    b_delete_all_shapes_in_layer(b, ref(b, layer).value)
  end

#=
Materials
A shape can be directly associated to a material or the shape can be associated
to a layer and the layer is then associated to the material
=#
@defproxy(material, Proxy, layer::Layer=current_layer(), data::BackendParameter=BackendParameter())
material(name::String, color::RGB=rgb(1,1,1), bvs...) = material(layer(name, true, color), BackendParameter(bvs...))
# Some backends prefer to use layers instead of materials
export material_as_layer, with_material_as_layer
const material_as_layer = Parameter(false)
use_material_as_layer(b::Backend) = material_as_layer()
with_material_as_layer(f::Function, b::Backend, m::Material) =
  use_material_as_layer(b) ?
    let cur_layer = b_current_layer(b),
        new_layer = ref(b, m.layer).value
      cur_layer == new_layer ?
        f() :
        begin
          b_current_layer(b, new_layer)
          let res = f()
            b_current_layer(b, cur_layer)
            res
          end
        end
      end :
    f()

realize(b::Backend, m::Material) =
  b_get_material(b, m.data(b))

# For compatibility
export set_material
const set_material = set_on!
# To facilitate accessing the material reference that is provided to the backends:
material_ref(b::Backend, m::Material) = ref(b, m).value
material_ref(b::Backend, s::Shape) = material_ref(b, s.material)

# These are pre-defined materials that need to be specified by each backend.
export material_point, material_curve, material_surface,
       material_basic, material_glass,
       material_metal, material_wood,
       material_concrete, material_plaster,
       material_grass, material_clay

const material_point = material("Points")
const material_curve = material("Curves")
const material_surface = material("Surfaces")
const material_basic = material("Basic")
const material_glass = material("Glass")
const material_metal = material("Metal")
const material_wood = material("Wood")
const material_concrete = material("Concrete")
const material_plaster = material("Plaster")
const material_grass = material("Grass")
const material_clay = material("Clay")

export default_point_material, default_curve_material, default_surface_material, default_material
const default_point_material = Parameter{Material}(material_point)
const default_curve_material = Parameter{Material}(material_curve)
const default_surface_material = Parameter{Material}(material_surface)
const default_material = Parameter{Material}(material_basic)

abstract type Shape0D <: Shape end
abstract type Shape1D <: Shape end
abstract type Shape2D <: Shape end
abstract type Shape3D <: Shape end

is_curve(s::Shape) = false
is_surface(s::Shape) = false
is_solid(s::Shape) = false

is_curve(s::Shape1D) = true
is_surface(s::Shape2D) = true
is_solid(s::Shape3D) = true

# To handle the direct use of shapes as paths:
b_stroke(b::Backend, path::Shape, mat) =
  and_mark_deleted(b, ref(b, path).value, path)

# HACK: Fix element type
Shapes0D = Vector{<:Any}
Shapes1D = Vector{<:Any}
Shapes2D = Vector{<:Any}

# This might be usable, so
export @defproxy, realize, Shape0D, Shape1D, Shape2D, Shape3D, void_ref

@defproxy(empty_shape, Shape0D)
@defproxy(universal_shape, Shape3D)
realize(b::Backend, s::EmptyShape) = void_ref(b)
realize(b::Backend, s::UniversalShape) = void_ref(b)


macro defshape(supertype, name_typename, fields...)
  # Merge this with defproxy
  (name, typename) = name_typename isa Symbol ?
    (name_typename, Symbol(string(map(uppercasefirst,split(string(name_typename),'_'))...))) :
    name_typename.args
  field_names = map(field -> field.args[1].args[1], fields)
  default_material =
    supertype == :Shape0D ? :default_point_material :
    supertype == :Shape1D ? :default_curve_material :
    supertype == :Shape2D ? :default_surface_material :
    supertype == :Shape3D ? :default_material :
    error("Unknown supertype:", supertype)
  esc(quote
    @defproxy($(name_typename), $(supertype), $(fields...), material::Material=$(default_material)())
    realize(b::Backend, s::$(typename)) =
      $(Symbol(:b_, name))(b, $(map(f->:(getproperty(s, $(QuoteNode(f)))), field_names)...), material_ref(b, s))
  end)
end

used_materials(s::Shape) = (s.material, )


@defshape(Shape0D, point, position::Loc=u0())

@defshape(Shape1D, line, vertices::Locs=[u0(), ux()])
line(v0::Loc, v1::Loc, vs...) = line([v0, v1, vs...])
@defshape(Shape1D, closed_line, vertices::Locs=[u0(), ux(), uy()])
closed_line(v0::Loc, v1::Loc, vs...) = closed_line([v0, v1, vs...])
@defshape(Shape1D, spline, points::Locs=[u0(), ux(), uy()], v0::Union{Bool,Vec}=false, v1::Union{Bool,Vec}=false)
spline(v0::Loc, v1::Loc, vs...) = spline([v0, v1, vs...])

#=
evaluate(s::Spline, t::Real) =
  let interpolator = s.interpolator
    if ismissing(interpolator())
      interpolator(curve_interpolator(s.points))
    end
    let p = interpolator()(t),
        vt = Interpolations.gradient(interpolator(), t)[1],
        vn = Interpolations.hessian(interpolator(), t)[1]
      loc_from_o_vx_vy(
        xyz(p[1], p[2], p[3], world_cs),
        vxyz(vt[1], vt[2], vt[3], world_cs),
        vxyz(vn[1], vn[2], vn[3], world_cs))
    end
  end

evaluate(s::Spline, t::Real) =
  let interpolator = s.interpolator
    if ismissing(interpolator())
      interpolator(curve_interpolator(s.points))
    end
    let p = interpolator()(t),
        vt = Interpolations.gradient(interpolator(), t)[1],
        vn = Interpolations.hessian(interpolator(), t)[1],
        vy = cross(vt, vn)
      loc_from_o_vx_vy(
        xyz(p[1], p[2], p[3], world_cs),
        vxyz(vn[1], vn[2], vn[3], world_cs),
        vxyz(vy[1], vy[2], vy[3], world_cs))
    end
  end
=#
map_division(f::Function, s::Spline, n::Int, backend::Backend=backend(s)) =
  backend_map_division(backend, f, s, n)
#=HACK, THIS IS NOT READY, YET. COMPARE WITH THE BACKEND VERSION!!!!!!
  let (t1, t2) = curve_domain(s)
    map_division(t1, t2, n) do t
        f(frame_at(s, t))
    end
  end
=#
#(def-base-shape 1D-shape (spline* [pts : (Listof Loc) (list (u0) (ux) (uy))] [v0 : (U Boolean Vec) #f] [v1 : (U Boolean Vec) #f]))

@defshape(Shape1D, closed_spline, points::Locs=[u0(), ux(), uy()])
closed_spline(v0, v1, vs...) = closed_spline([v0, v1, vs...])
@defshape(Shape1D, circle, center::Loc=u0(), radius::Real=1)
@defshape(Shape1D, arc, center::Loc=u0(), radius::Real=1, start_angle::Real=0, amplitude::Real=pi)
@defshape(Shape1D, elliptic_arc, center::Loc=u0(), radius_x::Real=1, radius_y::Real=1, start_angle::Real=0, amplitude::Real=pi)
@defshape(Shape1D, ellipse, center::Loc=u0(), radius_x::Real=1, radius_y::Real=1)
@defshape(Shape1D, polygon, vertices::Locs=[u0(), ux(), uy()])
polygon(v0, v1, vs...) = polygon([v0, v1, vs...])
@defshape(Shape1D, regular_polygon, edges::Integer=3, center::Loc=u0(), radius::Real=1, angle::Real=0, inscribed::Bool=true)
@defshape(Shape1D, rectangle, corner::Loc=u0(), dx::Real=1, dy::Real=1)
rectangle(p::Loc, q::Loc) =
  let v = in_cs(q - p, p.cs)
    rectangle(p, v.x, v.y)
  end

#
#@defshape dimension(p0::Loc, p1::Loc, p::Loc, scale::Real, style::Symbol)
#@defshape dimension(p0::Loc, p1::Loc, sep::Real, scale::Real, style::Symbol)
@defshape(Shape1D, dimension, from::Loc=u0(), to::Loc=ux(), text::AbstractString=string(distance(p0, p1)), size::Real=1)

# Surfaces

@defshape(Shape2D, surface_circle, center::Loc=u0(), radius::Real=1)
@defshape(Shape2D, surface_arc, center::Loc=u0(), radius::Real=1, start_angle::Real=0, amplitude::Real=pi)
@defshape(Shape2D, surface_elliptic_arc, center::Loc=u0(), radius_x::Real=1, radius_y::Real=1, start_angle::Real=0, amplitude::Real=pi)
@defshape(Shape2D, surface_ellipse, center::Loc=u0(), radius_x::Real=1, radius_y::Real=1)
@defshape(Shape2D, surface_polygon, vertices::Locs=[u0(), ux(), uy()])
surface_polygon(v0, v1, vs...) = surface_polygon([v0, v1, vs...])
@defshape(Shape2D, surface_regular_polygon, edges::Integer=3, center::Loc=u0(), radius::Real=1, angle::Real=0, inscribed::Bool=true)
@defshape(Shape2D, surface_rectangle, corner::Loc=u0(), dx::Real=1, dy::Real=1)
surface_rectangle(p::Loc, q::Loc) =
  let v = in_cs(q - p, p.cs)
    surface_rectangle(p, v.x, v.y)
  end

@defproxy(surface, Shape2D, frontier::Shapes1D=[circle()])
surface(c0::Shape, cs...) = surface([c0, cs...])
#To be removed
surface_from = surface

@defproxy(surface_path, Shape2D, path::ClosedPath=[circular_path()])
realize(b::Backend, s::SurfacePath) =
  backend_fill(b, s.path)

surface_boundary(s::Shape2D, backend::Backend=top_backend()) =
  backend_surface_boundary(backend, s)

curve_domain(s::Shape1D, backend::Backend=top_backend()) =
  backend_curve_domain(backend, s)
map_division(f::Function, s::Shape1D, n::Int, backend::Backend=top_backend()) =
  backend_map_division(backend, f, s, n)

surface_domain(s::Shape2D, backend::Backend=top_backend()) =
  backend_surface_domain(backend, s)
map_division(f::Function, s::Shape2D, nu::Int, nv::Int, backend::Backend=top_backend()) =
  backend_map_division(backend, f, s, nu, nv)


path_vertices(s::Shape1D) = path_vertices(shape_path(s))
shape_path(s::Circle) = circular_path(s.center, s.radius)
shape_path(s::Spline) = open_spline_path(s.points, s.v0, s.v1)
shape_path(s::ClosedSpline) = closed_spline_path(s.points)

@defshape(Shape0D, text, str::String="", corner::Loc=u0(), height::Real=1)

export text_centered
text_centered(str::String="", center::Loc=u0(), height::Real=1) =
  text(str, add_xy(center, -length(str)*height*0.85/2, -height/2), height)

# This is for unknown shapes (they are opaque, the only thing you can do with then
# might be just delete them)
@defproxy(unknown, Shape3D, baseref::Any=required())

@defshape(Shape3D, sphere, center::Loc=u0(), radius::Real=1)

@defshape(Shape3D, torus, center::Loc=u0(), re::Real=1, ri::Real=1/2)
@defshape(Shape3D, cuboid,
  b0::Loc=u0(),        b1::Loc=add_x(b0,1), b2::Loc=add_y(b1,1), b3::Loc=add_x(b2,-1),
  t0::Loc=add_z(b0,1), t1::Loc=add_x(t0,1), t2::Loc=add_y(t1,1), t3::Loc=add_x(t2,-1))

@defshape(Shape3D, regular_pyramid_frustum, edges::Integer=4, cb::Loc=u0(), rb::Real=1, angle::Real=0, h::Real=1, rt::Real=1, inscribed::Bool=true)
regular_pyramid_frustum(edges::Integer, cb::Loc, rb::Real, angle::Real, ct::Loc, rt::Real=1, inscribed::Bool=true) =
  let (c, h) = position_and_height(cb, ct)
    regular_pyramid_frustum(edges, c, rb, angle, h, rt, inscribed)
  end

@defshape(Shape3D, regular_pyramid, edges::Integer=3, cb::Loc=u0(), rb::Real=1, angle::Real=0, h::Real=1, inscribed::Bool=true)
regular_pyramid(edges::Integer, cb::Loc, rb::Real, angle::Real, ct::Loc, inscribed::Bool=true) =
  let (c, h) = position_and_height(cb, ct)
    regular_pyramid(edges, c, rb, angle, h, inscribed)
  end

@defshape(Shape3D, pyramid_frustum, bs::Locs=[ux(), uy(), uxy()], ts::Locs=[uxz(), uyz(), uxyz()])

@defshape(Shape3D, pyramid, bs::Locs=[ux(), uy(), uxy()], t::Loc=uz())

@defshape(Shape3D, regular_prism, edges::Integer=3, cb::Loc=u0(), r::Real=1, angle::Real=0, h::Real=1, inscribed::Bool=true)
regular_prism(edges::Integer, cb::Loc, r::Real, angle::Real, ct::Loc, inscribed::Bool=true) =
  let (c, h) = position_and_height(cb, ct)
    regular_prism(edges, c, r, angle, h, inscribed)
  end

@defshape(Shape3D, prism, bs::Locs=[ux(), uy(), uxy()], v::Vec=vz(1))
prism(bs::Locs, h::Real) =
  prism(bs, vz(h))

@defshape(Shape3D, right_cuboid, cb::Loc=u0(), width::Real=1, height::Real=1, h::Real=1)
right_cuboid(cb::Loc, width::Real, height::Real, ct::Loc, angle::Real=0) =
  let (c, h) = position_and_height(cb, ct),
      o = angle == 0 ? c : loc_from_o_phi(c, angle)
    right_cuboid(o, width, height, h)
  end

@defshape(Shape3D, box, c::Loc=u0(), dx::Real=1, dy::Real=dx, dz::Real=dy)
box(c0::Loc, c1::Loc, others...) =
  let v = in_cs(c1, c0)-c0
    if v.x < 0
      c0 = add_x(c0, v.x)
    end
    if v.y < 0
      c0 = add_y(c0, v.y)
    end
    if v.z < 0
      c0 = add_z(c0, v.z)
    end
    box(c0, abs(v.x), abs(v.y), abs(v.z), others...)
  end

@defshape(Shape3D, cone, cb::Loc=u0(), r::Real=1, h::Real=1)
cone(cb::Loc, r::Real, ct::Loc) =
  let (c, h) = position_and_height(cb, ct)
    cone(c, r, h)
  end
@defshape(Shape3D, cone_frustum, cb::Loc=u0(), rb::Real=1, h::Real=1, rt::Real=1)
cone_frustum(cb::Loc, rb::Real, ct::Loc, rt::Real; material=default_material()) =
  let (c, h) = position_and_height(cb, ct)
    cone_frustum(c, rb, h, rt, material)
  end
@defshape(Shape3D, cylinder, cb::Loc=u0(), r::Real=1, h::Real=1)
cylinder(cb::Loc, r::Real, ct::Loc; material=default_material()) =
  let (c, h) = position_and_height(cb, ct)
    cylinder(c, r, h, material)
  end

#=
An isosurface is surface that is described by the implícit equation

F(x,y,z) = k

It is frequent to use the simpler form

G(x,y,z) = 0,

by defining G(x,y,z) = F(x,y,z) - k

The name 'iso' means 'same value', which comes from the fact that F(x,y,z) has
always the same value. The idea is that we sample all points in space, applying
F to each one, and we those where F returns k (or G returns zero) belong to the
isosurface. There are several algorithms that speed up this sampling process,
being the marching cubes the most popular one.
=#

@defshape(Shape3D, isosurface, frep::Function=loc->sph_rho(loc), bounding_box::Locs=[xyz(-1,-1,-1), xyz(+1,+1,+1)])

@defshape(Shape3D, extrusion, profile::Union{Shape,Path,Region}=circular_path(), v::Vec=vz(1), cb::Loc=u0())
extrusion(profile, h::Real) =
  extrusion(profile, vz(h))

@defshape(Shape3D, sweep, path::Union{Shape1D, Path}=circle(), profile::Union{Shape,Path,Region}=point(), rotation::Real=0, scale::Real=1)

@defproxy(revolve_point, Shape1D, profile::Shape0D=point(), p::Loc=u0(), n::Vec=vz(1,p.cs), start_angle::Real=0, amplitude::Real=2*pi)
@defproxy(revolve_curve, Shape2D, profile::Shape1D=line(), p::Loc=u0(), n::Vec=vz(1,p.cs), start_angle::Real=0, amplitude::Real=2*pi)
@defproxy(revolve_surface, Shape3D, profile::Shape2D=circle(), p::Loc=u0(), n::Vec=vz(1,p.cs), start_angle::Real=0, amplitude::Real=2*pi)
revolve(profile::Shape=point(x(1)), p::Loc=u0(), n::Vec=vz(1,p.cs), start_angle::Real=0, amplitude::Real=2*pi) =
  if is_point(profile)
    revolve_point(profile, p, n, start_angle, amplitude)
  elseif is_curve(profile)
    revolve_curve(profile, p, n, start_angle, amplitude)
  elseif is_surface(profile)
    revolve_surface(profile, p, n, start_angle, amplitude)
  elseif is_union_shape(profile)
    union(map(s->revolve(s, p, n, start_angle, amplitude), profile.shapes))
  elseif is_empty_shape(profile)
    profile
  else
    error("Profile is neither a point nor a curve nor a surface")
  end

backend_revolve_point(b::Backend, profile::Shape, p::Loc, n::Vec, start_angle::Real, amplitude::Real) = error("Finish this")
backend_revolve_curve(b::Backend, profile::Shape, p::Loc, n::Vec, start_angle::Real, amplitude::Real) = error("Finish this")
backend_revolve_surface(b::Backend, profile::Shape, p::Loc, n::Vec, start_angle::Real, amplitude::Real) = error("Finish this")

realize(b::Backend, s::RevolvePoint) =
  backend_revolve_point(b, s.profile, s.p, s.n, s.start_angle, s.amplitude)
realize(b::Backend, s::RevolveCurve) =
  backend_revolve_curve(b, s.profile, s.p, s.n, s.start_angle, s.amplitude)
realize(b::Backend, s::RevolveSurface) =
  backend_revolve_surface(b, s.profile, s.p, s.n, s.start_angle, s.amplitude)

@defproxy(loft_points, Shape1D, profiles::Shapes0D=Shape[], rails::Shapes=Shape[], ruled::Bool=false, closed::Bool=false)
@defproxy(loft_curves, Shape2D, profiles::Shapes1D=Shape[], rails::Shapes=Shape[], ruled::Bool=false, closed::Bool=false)
@defproxy(loft_surfaces, Shape3D, profiles::Shapes2D=Shape[], rails::Shapes=Shape[], ruled::Bool=false, closed::Bool=false)
@defproxy(loft_curve_point, Shape2D, profile::Shape1D=circle(), point::Shape0D=point(z(1)))
@defproxy(loft_surface_point, Shape3D, profile::Shape2D=surface_circle(), point::Shapes=point(z(1)))

loft(profiles::Shapes=Shape[], rails::Shapes=Shape[], ruled::Bool=false, closed::Bool=false) =
  if all(is_point, profiles)
    loft_points(profiles, rails, ruled, closed)
  elseif all(is_curve, profiles)
    loft_curves(profiles, rails, ruled, closed)
  elseif all(is_surface, profiles)
    loft_surfaces(profiles, rails, ruled, closed)
  elseif length(profiles) == 2
    let (p, sh) = if is_point(profiles[1])
                    (profiles[1], profiles[2])
                  elseif is_point(profiles[2])
                    (profiles[2], profiles[1])
                  else
                    error("Cross sections are neither points nor curves nor surfaces")
                  end
      if is_curve(sh)
        loft_curve_point(sh, p)
      elseif is_surface(sh)
        loft_surface_point(sh, p)
      else
        error("Can't loft the shapes")
      end
    end
  else
    error("Cross sections are neither points nor curves nor surfaces")
  end

loft_ruled(profiles::Shapes=Shape[]) = loft(profiles, Shape[], true, false)
export loft, loft_ruled

realize(b::Backend, s::LoftPoints) = backend_loft_points(backend(s), s.profiles, s.rails, s.ruled, s.closed)
realize(b::Backend, s::LoftCurves) = backend_loft_curves(backend(s), s.profiles, s.rails, s.ruled, s.closed)
realize(b::Backend, s::LoftSurfaces) = backend_loft_surfaces(backend(s), s.profiles, s.rails, s.ruled, s.closed)
realize(b::Backend, s::LoftCurvePoint) = backend_loft_curve_point(backend(s), s.profile, s.point)
realize(b::Backend, s::LoftSurfacePoint) = backend_loft_surface_point(backend(s), s.profile, s.point)

backend_loft_points(b::Backend, profiles::Shapes, rails::Shapes, ruled::Bool, closed::Bool) =
  let f = (ruled ? (closed ? polygon : line) : (closed ? closed_spline : spline))
    and_delete_shapes(ref(b, f(map(point_position, profiles), backend=b)),
                      vcat(profiles, rails))
  end

@defproxy(move, Shape3D, shape::Shape=point(), v::Vec=vx())
@defproxy(scale, Shape3D, shape::Shape=point(), s::Real=1, p::Loc=u0())
@defproxy(rotate, Shape3D, shape::Shape=point(), angle::Real=0, p::Loc=u0(), v::Vec=vz(1,p.cs))
@defproxy(transform, Shape3D, shape::Shape=point(), xform::Loc=u0())

#####################################################################

# We can also translate some shapes
translate(s::Line, v::Vec) = line(map(p -> p+v, s.vertices))
translate(s::Polygon, v::Vec) = polygon(map(p -> p+v, s.vertices))
translate(s::Circle, v::Vec) = circle(s.center+v, s.radius)
translate(s::Text, v::Vec) = text(s.str, s.c+v, s.h)

# We can translate arrays of Shapes
translate(ss::Shapes, v::Vec) = translate.(ss, v)

# We can compute the length of shapes as long as we can convert them
curve_length(s::Shape) = curve_length(convert(Path, s))

# We will also need to compute a bounding rectangle
bounding_rectangle(s::Union{Line, Polygon}) =
    bounding_rectangle(s.vertices)

bounding_rectangle(pts::Locs) =
    let min_p = pts[1]
        max_p = min_p
        for i in 2:length(pts)
            min_p = min_loc(min_p, pts[i])
            max_p = max_loc(max_p, pts[i])
        end
        [min_p, max_p]
    end

bounding_rectangle(ss::Shapes) =
    bounding_rectangle(mapreduce(bounding_rectangle, vcat, ss))


#####################################################################

#####################################################################
## Conversions

convert(::Type{Path}, s::Rectangle) =
  and_delete_shape(rectangular_path(s.corner, s.dx, s.dy), s)
convert(::Type{Path}, s::Line) =
  and_delete_shape(convert(OpenPath, s.vertices), s)
convert(::Type{Path}, s::Circle) =
  and_delete_shape(circular_path(s.center, s.radius), s)
convert(::Type{Path}, s::Polygon) =
  and_delete_shape(polygonal_path(s.vertices), s)
convert(::Type{Region}, s::SurfaceCircle) =
  and_delete_shape(region(circular_path(s.center, s.radius)), s)
convert(::Type{Region}, s::SurfacePolygon) =
  and_delete_shape(region(polygonal_path(s.vertices), s))
#####################################################################
## Paths can be used to generate surfaces and solids

@defproxy(sweep_path, Shape3D, path::Path=polygonal_path(), profile::Path=circular_path(), rotation::Real=0, scale::Real=1)


#####################################################################
export curve_domain, surface_domain, frame_at
surface_domain(s::SurfaceRectangle) = (0, s.dx, 0, s.dy)
surface_domain(s::SurfaceCircle) = (0, s.radius, 0, 2pi)
surface_domain(s::SurfaceArc) = (0, s.radius, s.start_angle, s.amplitude)

export backend_frame_at
backend_frame_at(b::Backend, s::Shape2D, u::Real, v::Real) = error("BUM")

frame_at(c::Shape1D, t::Real) = backend_frame_at(backend(c), c, t)
frame_at(s::Shape2D, u::Real, v::Real) = backend_frame_at(backend(s), s, u, v)

#Some specific cases can be handled in an uniform way without the backend
frame_at(s::SurfaceRectangle, u::Real, v::Real) = add_xy(s.corner, u, v)
frame_at(s::SurfaceCircle, u::Real, v::Real) = add_pol(s.center, u, v)

export union, intersection, subtraction
#=
We do some pre-filtering to deal with the presence of empty shapes or to simplify one-arg cases.
=#

@defproxy(union_shape, Shape3D, shapes::Shapes=Shape[])
union(shapes::Shapes) =
  let non_empty_shapes = filter(s -> !is_empty_shape(s), shapes),
      count_non_empty_shapes = length(non_empty_shapes)
    count_non_empty_shapes == 0 ? empty_shape() :
    count_non_empty_shapes == 1 ? non_empty_shapes[1] :
    union_shape(non_empty_shapes)
  end

union(shape::Shape, shapes...) = union([shape, shapes...])

@defproxy(intersection_shape, Shape3D, shapes::Shapes=Shape[])
intersection(shapes::Shapes) = intersection_shape(shapes)
intersection(shape::Shape, shapes...) =
  is_empty_shape(shape) || any(is_empty_shape, shapes) ? empty_shape() :
  shapes == [] ? shape : intersection_shape([shape, shapes...])

@defproxy(subtraction_shape2D, Shape2D, shape::Shape=surface_circle(), shapes::Shapes=Shape[])
@defproxy(subtraction_shape3D, Shape3D, shape::Shape=surface_sphere(), shapes::Shapes=Shape[])
subtraction(shape::Shape2D, shapes...) =
  is_empty_shape(shape) ? empty_shape() :
    let non_empty_shapes = filter(s -> !is_empty_shape(s), shapes),
        count_non_empty_shapes = length(non_empty_shapes)
      count_non_empty_shapes == 0 ? shape : subtraction_shape2D(shape, [non_empty_shapes...])
    end
subtraction(shape::Shape3D, shapes...) =
  is_empty_shape(shape) ? empty_shape() :
    let non_empty_shapes = filter(s -> !is_empty_shape(s), shapes),
        count_non_empty_shapes = length(non_empty_shapes)
      count_non_empty_shapes == 0 ? shape : subtraction_shape3D(shape, [non_empty_shapes...])
    end

@defproxy(slice, Shape3D, shape::Shape=sphere(), p::Loc=u0(), n::Vec=vz(1))

@defproxy(mirror, Shape3D, shape::Shape=sphere(), p::Loc=u0(), n::Vec=vz(1))
@defproxy(union_mirror, Shape3D, shape::Shape=sphere(), p::Loc=u0(), n::Vec=vz(1))

@defshape(Shape2D, surface_grid, points::Matrix{<:Loc}=zeros(Loc,(2,2)),
          closed_u::Bool=false, closed_v::Bool=false,
          smooth_u::Bool=true, smooth_v::Bool=true)

surface_grid(_points::Vector{<:Vector{<:Loc}},
  _closed_u=false, _closed_v=false, _smooth_u=true, _smooth_v=true, _material=default_material();
  points=_points, closed_u=_closed_u, closed_v=_closed_v, smooth_u=_smooth_u, smooth_v=_smooth_v, material=_material) =
  surface_grid(permutedims(hcat(points...)), closed_u, closed_v, smooth_u, smooth_v, material=material)
# For interpolator to work, we need this:

convert(::Type{Matrix{XYZ}}, ptss::Vector{Vector{<:Loc}}) =
  permutedims(hcat(ptss...))

surface_domain(s::SurfaceGrid) = (0.0, 1.0, 0.0, 1.0)
frame_at(s::SurfaceGrid, u::Real, v::Real) = evaluate(s, u, v)
map_division(f::Function, s::SurfaceGrid, nu::Int, nv::Int, backend::Backend=top_backend()) =
  let (u1, u2, v1, v2) = surface_domain(s)
    map_division(u1, u2, nu) do u
      map_division(v1, v2, nv) do v
        f(frame_at(s, u, v))
      end
    end
  end

@defshape(Shape2D, surface_mesh, vertices::Locs=[u0(), ux(), uy()], faces::Vector{Vector{Int}}=[[0,1,2]])

@defproxy(parametric_surface, Shape2D, definition::Function=(u,v)->xyz(u,v,0),
          domain_u::Tuple{Real,Real}=(0,1), domain_v::Tuple{Real,Real}=(0,1))

@defproxy(thicken, Shape3D, shape::Shape=surface_circle(), thickness::Real=1)

# Blocks

@defproxy(block, Shape, name::String="Block", shapes::Shapes = Shape[])
@defproxy(block_instance, Shape, block::Block=required(), loc::Loc=u0(), scale::Real=1.0)

################################################################################


################################################################################
bounding_box(shape::Shape) =
  bounding_box([shape])

bounding_box(shapes::Shapes=Shape[]) =
  if isempty(shapes)
    [u0(), u0()]
  else
    backend_bounding_box(backend(shapes[1]), shapes)
  end

backend_bounding_box(backend::Backend, shape::Shape) =
  throw(UndefinedBackendException())

@defcbs delete_shape(s::Shape)

b_delete_shape(b::Backend, s::Shape) =
  if realized(b, s)
    b_delete_refs(b, collect_ref(b, ref(b, s)))
    reset_ref(b, s)
  end

delete_shapes(ss::Shapes=Shape[], bs=current_backends()) =
  for s in ss
    delete_shape(s, bs)
  end

export and_delete_shape, and_delete_shapes, and_mark_deleted
and_delete_shape(r::Any, shape::Shape) =
  begin
    delete_shape(shape)
    r
  end

and_delete_shapes(r::Any, shapes::Shapes) =
  begin
    delete_shapes(shapes)
    r
  end

and_mark_deleted(b::Backend, r::Any, shape) =
  begin
    mark_deleted(b, shape)
    r
  end

# Common implementations for realize function

realize(b::Backend, s::UnionShape) =
  unite_refs(b, UnionRef(tuple(map(s->ref(b, s), s.shapes)...)))

realize(b::Backend, s::Union{SubtractionShape2D,SubtractionShape3D}) =
    subtract_ref(b, ref(b, s.shape), unite_refs(b, UnionRef(tuple(map(s->ref(b, s), s.shapes)...))))

function startSketchup(port)
  ENV["ROSETTAPORT"] = port
  args = "C:\\Users\\aml\\Dropbox\\AML\\Projects\\rosetta\\sketchup\\rosetta.rb"
  println(args)
  run(`cmd /C Sketchup -RubyStartup $args`)
  #Start listening for Sketchup
  listener = listen(port)
  connection = listener.accept()
  readline(connection) == "connected" ? connection : error("Could not connect!")
end

# CAD

@defcb select_position(prompt::String="Select a position")
@defcb select_positions(prompt::String="Select positions")
@defcb select_point(prompt::String="Select a point")
@defcb select_points(prompt::String="Select points")
@defcb select_curve(prompt::String="Select a curve")
@defcb select_curves(prompt::String="Select curves")
@defcb select_surface(prompt::String="Select a surface")
@defcb select_surfaces(prompt::String="Select surfaces")
@defcb select_solid(prompt::String="Select a solid")
@defcb select_solids(prompt::String="Select solids")
@defcb select_shape(prompt::String="Select a shape")
@defcb select_shapes(prompt::String="Select shapes")
@defshapeop register_shape_for_changes(s::Shape)
@defshapeop unregister_shape_for_changes(s::Shape)
@defshapeop waiting_for_changes()
@defcb changed_shape(shapes::Shapes)

@defcbs highlight_shape(s::Shape)
b_highlight_shape(b::Backend, s::Shape) =
  if realized(b, s)
    b_highlight_refs(b, collect_ref(b, ref(b, s)))
  end

export highlight_shapes
highlight_shapes(ss::Shapes=Shape[], bs=current_backends()) =
  for s in ss
    highlight_shape(s, bs)
  end

#
@defcbs unhighlight_shape(s::Shape)
b_unhighlight_shape(b::Backend, s::Shape) =
  if realized(b, s)
    b_unhighlight_refs(b, collect_ref(b, ref(b, s)))
  end

export unhighlight_shapes
unhighlight_shapes(ss::Shapes=Shape[], bs=current_backends()) =
  for s in ss
    unhighlight_shape(s, bs)
  end

export unhighlight_all_shapes
const unhighlight_all_shapes = unhighlight_all_refs


capture_shape(s=select_shape("Select shape to be captured")) =
  if ! isnothing(s)
    generate_captured_shape(s, backend(s))
  end

capture_shapes(ss=select_shapes("Select shapes to be captured")) =
    generate_captured_shapes(ss, backend(ss[1]))

export register_for_changes
register_for_changes(shapes::Shapes) =
  map(shapes) do shape
    register_shape_for_changes(shape, backend(shape))
  end

export unregister_for_changes
unregister_for_changes(shapes::Shapes) =
  map(shapes) do shape
    unregister_shape_for_changes(shape, backend(shape))
  end

waiting_for_changes(shapes::Shapes) =
  waiting_for_changes(shapes[1], backend(shapes[1]))

export on_change
on_change(f, shape::Shape) = on_change(f, [shape])
on_change(f, shapes) =
  let registered = register_for_changes(shapes)
    try
      while waiting_for_changes(shapes)
        let changed = changed_shape(shapes)
          f()
        end
      end
    finally
      unregister_for_changes(registered)
    end
  end

#
export with_shape_dependency
with_shape_dependency(f, ss) =
    let shapes = collecting_shapes() do
                    f()
                end
        on_change(ss) do
            try
                delete_shapes(shapes)
            catch e
            end
            shapes = collecting_shapes() do
                f()
            end
        end
    end

#
export internalize_shape, internalize_shapes

internalize_shape(s=select_shape("Select shape to be internalized")) =
  if ! isnothing(s)
    println(meta_program(s))
  end

internalize_shapes(ss=select_shapes("Select shapes to be internalized")) =
    println(meta_program(ss))


# Seletion
export select_one_with_prompt, select_many_with_prompt
select_one_with_prompt(prompt::String, b::Backend, f::Function) =
  let ans = select_many_with_prompt(prompt, b, f)
    length(ans) > 0 ? ans[1] : nothing
  end

select_many_with_prompt(prompt::String, b::Backend, f::Function) =
  begin
    @info "$(prompt) on the $(b) backend."
    map(id -> b_shape_from_ref(b, id), f(connection(b), prompt))
  end

export save_view
save_view(name::String="View") =
  let path = prepare_for_saving_file(render_pathname(name))
    save_view(path, top_backend())
    path
  end

@defcbs set_time_place(date::DateTime=DateTime(2020, 9, 21, 10, 0, 0), latitude::Real=39, longitude::Real=-9, elevation::Real=0, meridian::Real=0)
@defcbs set_sky(turbidity::Real=5, sun::Bool=true)
@defcbs set_ground(level::Real=0, material::Material=material_basic)
@defcbs realistic_sky(date::DateTime=DateTime(2020, 9, 21, 10, 0, 0), latitude::Real=39, longitude::Real=-9, elevation::Real=0, meridian::Real=0, turbidity::Real=5, sun::Bool=true)

export ground
ground(level::Real=0, material::Material=material_basic, backend::Backend=top_backend()) =
  b_set_ground(backend, level, material_ref(backend, material))


############################################################
# Analysis

abstract type Analysis end
abstract type StructuralAnalysis <: Analysis end
abstract type LightingAnalysis <: Analysis end

###########################################################
# Geometric properties

# Axis-aligned Bounding Box

# Centroid
export centroid
centroid(s::Sphere) = s.center
centroid(s::Cylinder) = add_z(s.cb, s.h/2)

###########################################################

export show_cs

show_cs(p, scale=1) =
    let rcyl = scale/10,
        rcon = scale/5,
        lcyl = scale,
        lcon = scale/5,
        px = add_x(p, 3*lcyl),
        py = add_y(p, 2*lcyl),
        pz = add_z(p, 1*lcyl)
      union(args...) = args[end] # Unity is having problems with unions
        union(cylinder(p, rcyl, px),
              cone(px, rcon, add_x(px, lcon)),
              cylinder(p, rcyl, py),
              cone(py, rcon, add_y(py, lcon)),
              cylinder(p, rcyl, pz),
              cone(pz, rcon, add_z(pz, lcon)))
    end

#
nonzero_offset(l::Line, d::Real) =
  line(offset(l.vertices, d, false))

#
export stroke, b_stroke
stroke(path;
    material::Material=default_curve_material(),
	  backend::Backend=top_backend(),
	  backends::Backends=(backend,)) =
  for backend in backends
    let mat = material_ref(backend, material)
      b_stroke(backend, path, mat)
    end
  end
export fill, b_fill
fill(path;
    material::Material=default_surface_material(),
    backend::Backend=top_backend(),
    backends::Backends=(backend,)) =
  for backend in backends
    let mat = material_ref(backend, material)
      b_fill(backend, path, mat)
    end
  end
