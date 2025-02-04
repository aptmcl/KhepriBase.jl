export GenericRef,
       NativeRef,
       NativeRefs,
       DynRef,
       DynRefs,
       void_ref,
       ensure_ref,
       map_ref,
       collect_ref
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
       create_layer,
       current_layer,
       delete_all_shapes_in_layer,
       merge_materials,
       connection,
       @deffamily,
       @defproxy,
       subpath,
       subpath_starting_at,
       subpath_ending_at,
       bounding_box,
       capture_shape, capture_shapes,
       extrusion, sweep, revolve, loft

#References can be (single or multiple) native references
abstract type GenericRef{K,T} end

# Typically, we have a reference to the represented object
struct NativeRef{K,T} <: GenericRef{K,T}
  value::T
end
# Sometimes, the represented object require multiple references
struct NativeRefs{K,T} <: GenericRef{K,T}
  values::Vector{T}
end

ensure_ref(b::Backend{K,T}, r::T) where {K,T} = NativeRef{K,T}(r)
ensure_ref(b::Backend{K,T}, rs::Vector{T1}) where {K,T,T1<:T} =
  length(rs) == 1 ?
    NativeRef{K,T}(rs[1]) :
    NativeRefs{K,T}(rs)
ensure_ref(b::Backend{K,T}, rs::Vector{Vector{T1}}) where {K,T,T1<:T} =
  ensure_ref(b, reduce(vcat, rs))

ensure_ref(b::Backend{K,T}, r::Vector{Any}) where {K,T} = begin
  @warn("Unexpected reference $r")
  ensure_ref(b, T[r...])
end
ensure_ref(b::Backend, r::Any) = error("Unexpected reference $r")

# currying
map_ref(b::Backend{K,T}, f::Function) where {K,T} = r -> map_ref(b, f, r)

map_ref(b::Backend{K,T}, f::Function, r::NativeRef{K,T}) where {K,T} = f(r.value)
map_ref(b::Backend{K,T}, f::Function, r::NativeRefs{K,T}) where {K,T} = map(f, r.values)

# currying
collect_ref(b::Backend{K,T}) where {K,T} = 
  (r::GenericRef{K,T}) -> collect_ref(b, r)

collect_ref(b::Backend{K,T}, r::NativeRef{K,T}) where {K,T} = T[r.value]
collect_ref(b::Backend{K,T}, r::NativeRefs{K,T}) where {K,T} = r.values

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
#=
macro curryable(def)
  @assert def.head === :(=)
  let name = def.args[1].args[1],
      params = def.args[1].args[2:end],
      body = def.args[2:end]
      defs = [:($name($(params[1:j])...) = $name($(params[i+1:]))))]
    quote
      $def
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

show(io::IO, s::Shape) =
  print(io, "$(typeof(s))(...)")

map_ref(f::Function, b::Backend, s::Shape) = map_ref(b, f, ref(b, s))
collect_ref(b::Backend, s::Shape) = collect_ref(b, ref(b, s))
collect_ref(b::Backend, ss::Shapes) = mapreduce(s->collect_ref(b, ref(b, s)), vcat, ss, init=[])

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
Even if a backend is eager, it might be necessary to temporarily delay the
realization of shapes, particularly, when the construction is incremental.
In that case, we collect all created shapes and then realize them at the end.
=#

delaying_realize = Parameter(false)
delayed_realizations = Parameter([])
with_transaction(fn) = 
  with(delayed_realizations, []) do
    with(delaying_realize, true) do
      fn()
    end
    for s in delayed_realizations()
      maybe_realize(s)
    end
  end

maybe_realize(s) =
  let backends = current_backends()
    delaying_realize() ?
      push!(delayed_realizations(), s) :
      for b in backends
        maybe_realize(b, s)
      end
  end

maybe_realize(::EagerRealization, b, s) =
  if ! realized(b, s)
    force_realize(b, s)
  end

maybe_realize(t::LazyRealization, b, s) =
  save_shape!(b, s)


abstract type LazyBackend{K,T} <: Backend{K,T} end
realization_type(::Type{<:LazyBackend}) = LazyRealization()


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
# We use a dict from shapes to file locations
# and a dict from file locations to shapes
shape_to_file_locations = IdDict()
file_location_to_shapes = Dict()

export traceability, trace_depth, 
       excluded_modules, clear_trace!, 
       shape_source, source_shapes, 
       shape_to_file_locations, file_location_to_shapes, 
       highlight_source_shapes

shape_source(s) = get(shape_to_file_locations, s, [])
source_shapes(file::Symbol, line) = get(file_location_to_shapes, (file, line), Shape[])
source_shapes(file::String, line) = source_shapes(Symbol(file), line)
highlight_source_shapes(file, line) =
  let shapes = source_shapes(file, line)
    unhighlight_all_shapes()
    highlight_shapes(shapes)
  end
# This is a poor's man approach to JSON
struct JSON
  sources
end
show(io::IO, json::JSON) =
  begin
    println(io, "[")
    for (i, src) in enumerate(json.sources)
      if i > 1
        print(io, ",")
      end
      print(io, "[", repr(string(src[1])), ",", src[2], "]")
    end
    print(io, "]")
  end
select_shape_sources_string() = begin
  unhighlight_all_shapes()
  let sh = select_shape()
    highlight_shape(sh)
    # A poor man's approach at JSON
    JSON(shape_source(sh))
  end
end

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
delete_all_shapes() = begin
  empty!(KhepriBase.shape_to_file_locations)
  empty!(KhepriBase.file_location_to_shapes)
  delete_all_annotations()
  delete_all_refs()
end

@defcb all_shapes()
b_all_shapes(b::Backend) =
  Shape[maybe_existing_shape_from_ref(b, r) for r in b_all_refs(b)]
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
  mk_param(name,typ,init) = Expr(:kw, name, init) #Expr(:kw, Expr(:(::), name, typ), init)
  opt_params = map(mk_param, field_renames, field_types, map(init -> replace_in(init, field_replacements), field_inits))
  key_params = map(mk_param, field_names, field_types, field_renames)
  constructor_name = esc(name)
  predicate_name = esc(Symbol("is_", name_str))
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
    @noinline $(constructor_name)($(opt_params...); $(key_params...), ref::DynRefs=DynRefs()) =
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

@defproxy(layer, Proxy, name::String="Layer", active::Bool=true, color::RGBA=rgba(1,1,1,1))
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
material(name::String, bvs...) = material(layer(name, true), BackendParameter(bvs...))
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
  b_get_material(b, m.layer, m.data(b))
# By default, the layer is ignored in the creation of materials, as only a few backends depend on that.
b_get_material(b::Backend, layer, spec) =
  b_get_material(b, spec)

export merge_materials, merge_backend_materials
merge_materials(materials...) =
  let name = join([material.layer.name for material in materials], "_"),
      newdata = IdDict{Backend,Any}()
    for material in reverse(materials)
      # Violating an abstraction barrier!
      for (b, v) in material.data.value
        newdata[b] = haskey(newdata, b) ? merge_backend_materials(b, v, newdata[b]) : v
      end
    end
    material(name, newdata...)
  end
merge_backend_materials(b::Backend, v1, v2) = v1

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

# To handle the direct use of shapes as paths:
b_stroke(b::Backend, path::Shape, mat) =
  and_mark_deleted(b, ref(b, path).value, path)


# This might be usable, so
export @defproxy, realize, Shape0D, Shape1D, Shape2D, Shape3D, void_ref

macro defshape(supertype, name_typename, fields...)
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
# The next two should be merged
@defshape(Shape1D, closed_line, vertices::Locs=[u0(), ux(), uy()])
closed_line(v0::Loc, v1::Loc, vs...) = closed_line([v0, v1, vs...])
@defshape(Shape1D, polygon, vertices::Locs=[u0(), ux(), uy()])
polygon(v0, v1, vs...) = polygon([v0, v1, vs...])
@defshape(Shape1D, spline, points::Locs=[u0(), ux(), uy()], v0::Union{Bool,Vec}=false, v1::Union{Bool,Vec}=false)
spline(v0::Loc, v1::Loc, vs...) = spline([v0, v1, vs...])
@defshape(Shape1D, closed_spline, points::Locs=[u0(), ux(), uy()])
closed_spline(v0, v1, vs...) = closed_spline([v0, v1, vs...])
@defshape(Shape1D, circle, center::Loc=u0(), radius::Real=1)
@defshape(Shape1D, arc, center::Loc=u0(), radius::Real=1, start_angle::Real=0, amplitude::Real=pi)
@defshape(Shape1D, elliptic_arc, center::Loc=u0(), radius_x::Real=1, radius_y::Real=1, start_angle::Real=0, amplitude::Real=pi)
@defshape(Shape1D, ellipse, center::Loc=u0(), radius_x::Real=1, radius_y::Real=1)
@defshape(Shape1D, regular_polygon, edges::Integer=3, center::Loc=u0(), radius::Real=1, angle::Real=0, inscribed::Bool=true)
@defshape(Shape1D, rectangle, corner::Loc=u0(), dx::Real=1, dy::Real=1)
rectangle(p::Loc, q::Loc) =
  let v = in_cs(q - p, p.cs)
    rectangle(p, v.x, v.y)
  end

#=
Drawings might be annotated with labels, dimensions, and other stuff

However:
1. Not all backends support, e.g., Julia strings (which are Unicode-base).
2. Sometimes we might want to use entities that generate different textual representations depending on the backend

The solution is to delay the conversion from text to the corresponding but specific backend data 
=#
# HACK: This needs to become a field of the backend. Each one knows about its annotations.
const annotations = Shape[]

export delete_all_annotations
delete_all_annotations() =
  begin
    for shape in annotations
      try
        delete_shape(shape)
      catch e
      end
    end
    empty!(annotations)
  end
add_annotation!(ann) =
  begin
    push!(annotations, ann)
    ann
  end

all_annotations() =
  annotations

prev_annotation_that(p) =
  let idx = findfirst(p, annotations)
    isnothing(idx) ?
      nothing :
      let ann = annotations[idx]
        # Don't reuse annotations that have different materials
        if true #ann.material.layer.color == default_material().layer.color
          try # might have been previously deleted, e.g., by delete_all_shapes()
            delete_shape(ann)
          catch e
            @warn("Was already deleted!")
          finally
            deleteat!(annotations, idx)
          end
          ann
        else
          nothing
        end
      end
  end


@defshape(Shape1D, dimension, from::Loc=u0(), to::Loc=ux(), text::Any=string(distance(from, to)), size::Real=1, offset::Real=0.1)
@defshape(Shape1D, arc_dimension, center::Loc=u0(), radius::Real=1, start_angle::Real=0, amplitude::Real=pi, radius_text::Any=string(radius), amplitude_text::Any=string(amplitude), size::Real=1, offset::Real=0.1)

@defshape(Shape0D, labels, p::Loc=u0(), data::Vector{@NamedTuple{txt::Any, mat::Material, scale::Real}}=[])

export default_annotation_material
const default_annotation_material = Parameter{Material}(material(layer("annotation", true, rgba(0, 0, 0.5, 1.0))))
export default_annotation_scale
const default_annotation_scale = Parameter{Real}(1.0)

existing_material(mat, mats) =
  any(m -> m.layer.color == mat.layer.color, mats)

equal_illustration_properties(i1, i2) =
  i1.txt == i2.txt && 
  i1.mat.layer.color == i2.mat.layer.color &&
  i1.scale == i2.scale

export label
label(p, txt, mat=default_annotation_material(), scale=default_annotation_scale()) =
  let ann = prev_annotation_that(ann->is_labels(ann) && isapprox(p, ann.p, atol=1e-3)),
      new_d = (txt=txt, mat=mat, scale=scale);
    add_annotation!(
      isnothing(ann) ?
        labels(p, [new_d]) :
        any(d->equal_illustration_properties(new_d, d), ann.data) ?
          labels(p, ann.data) : 
          labels(p, [ann.data..., new_d]))
  end

@defshape(Shape1D, vectors_illustration, start::Loc=u0(), angle::Real=0, 
  radii::Vector{Real}=[],
  radii_texts::Vector{Any}=[],
  mats::Vector{Material}=[])
export vector_illustration
vector_illustration(p, a, r, txt, mat=default_annotation_material()) =
  let ann = prev_annotation_that(ann->is_vectors_illustration(ann) && isequal(p, ann.start) && isequal(a, ann.angle))
    add_annotation!(
      isnothing(ann) ?
        vectors_illustration(p, a, [r], [txt], [mat]) :
        (txt, mat.layer.color) in zip(ann.radii_texts, [mat.layer.color for mat in ann.mats]) ?
          vectors_illustration(p, a, ann.radii, ann.radii_texts, ann.mats) :
          vectors_illustration(p, a, [ann.radii..., r], [ann.radii_texts..., txt], [ann.mats..., mat]))
  end

@defshape(Shape1D, radii_illustration, center::Loc=u0(), 
  radii::Vector{Real}=[],
  radii_texts::Vector{Any}=[], 
  mats::Vector{Material}=[])
export radius_illustration
radius_illustration(c, r, txt, mat=default_annotation_material()) =
  let ann = prev_annotation_that(ann->is_radii_illustration(ann) && isequal(c, ann.center))
    add_annotation!(
      isnothing(ann) ?
        radii_illustration(c, [r], [txt], [mat]) :
        (txt, mat.layer.color) in zip(ann.radii_texts, [mat.layer.color for mat in ann.mats]) ?
          radii_illustration(c, ann.radii, ann.radii_texts, ann.mats) :
          radii_illustration(c, [ann.radii..., r], [ann.radii_texts..., txt], [ann.mats..., mat]))
  end

@defshape(Shape1D, angles_illustration, center::Loc=u0(), 
  radii::Vector{Real}=[], 
  start_angles::Vector{Real}=[], 
  amplitudes::Vector{Real}=[],
  radii_texts::Vector{Any}=[],
  start_angles_texts::Vector{Any}=[],
  amplitudes_texts::Vector{Any}=[], 
  mats::Vector{Material}=[])
export angle_illustration
angle_illustration(c, r, s, a, r_txt, s_txt, a_txt, mat=default_annotation_material()) =
  let ann = prev_annotation_that(ann -> is_angles_illustration(ann) && isequal(c, ann.center))
    add_annotation!(
      isnothing(ann) ?
        angles_illustration(c, [r], [s], [a], [r_txt], [s_txt], [a_txt], [mat]) :
        angles_illustration(c, [ann.radii..., r], 
                               [ann.start_angles..., s],
                               [ann.amplitudes..., a],
                               [ann.radii_texts..., r_txt], 
                               [ann.start_angles_texts..., s_txt], 
                               [ann.amplitudes_texts..., a_txt],
                               [ann.mats..., mat]))
  end

# This is similar to an angles illustration but not entirely equal
@defshape(Shape1D, arcs_illustration, center::Loc=u0(),
  radii::Vector{Real}=[],
  start_angles::Vector{Real}=[],
  amplitudes::Vector{Real}=[],
  radii_texts::Vector{Any}=[],
  start_angles_texts::Vector{Any}=[],
  amplitudes_texts::Vector{Any}=[],
  mats::Vector{Material}=[])
export arc_illustration
arc_illustration(c, r, s, a, r_txt, s_txt, a_txt, mat=default_annotation_material()) =
  let ann = prev_annotation_that(ann -> is_arcs_illustration(ann) && isequal(c, ann.center))
    add_annotation!(
      isnothing(ann) ?
        arcs_illustration(c, [r], [s], [a], [r_txt], [s_txt], [a_txt], [mat]) :
        arcs_illustration(c, [ann.radii..., r],
                             [ann.start_angles..., s],
                             [ann.amplitudes..., a],
                             [ann.radii_texts..., r_txt], 
                             [ann.start_angles_texts..., s_txt], 
                             [ann.amplitudes_texts..., a_txt], 
                             [ann.mats..., mat]))
  end

#=@defshape(Shape1D, line_illustration, vertices::Locs=[u0(), ux()],
  vertices_texts::Vector{AbstractString}=map(string, vertices))
@defshape(Shape1D, regular_polygon_illustration, edges::Integer=3, center::Loc=u0(), radius::Real=1, angle::Real=0, inscribed::Bool=true,
  center_text::AbstractString=string(center),
  radius_text::AbstractString=string(radius),
  angle_text::AbstractString=string(angle))
=#


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

@defshape(Shape2D, surface, frontier::Shapes1D=[circle()])
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
map_division(f::Function, s::Shape1D, n::Int) =
  b_map_division(backend(s), f, s, n)

surface_domain(s::Shape2D, backend::Backend=top_backend()) =
  backend_surface_domain(backend, s)
map_division(f::Function, s::Shape2D, nu::Int, nv::Int) =
  b_map_division(backend(s), f, s, nu, nv)

b_map_division(b::Backend, f::Function, s::Shape1D, n::Int) =
  map_division(f, shape_path(s), n)

#=
b_map_division(b::Backend, f::Function, s::Shape2D, nu::Int, nv::Int) =
  map_division(f, )
  #@bdef map_division(f::Function, s::SurfaceGrid, nu::Int, nv::Int)
=#

path_vertices(s::Shape1D) = path_vertices(shape_path(s))
path_frames(s::Shape1D) = path_frames(shape_path(s))

## shape_path takes a 0D/1D/2D shape and computes an equivalent path
shape_path(s::Point) = point_path(s.position)
shape_path(s::Circle) = circular_path(s.center, s.radius)
shape_path(s::Ellipse) = elliptic_path(s.center, s.radius_x, s.radius_y)
shape_path(s::Rectangle) = rectangular_path(s.corner, s.dx, s.dy)
shape_path(s::Line) = open_polygonal_path(s.vertices)
# The next two should be merged
shape_path(s::ClosedLine) = closed_polygonal_path(s.vertices)
shape_path(s::Polygon) = closed_polygonal_path(s.vertices)
shape_path(s::Spline) =
  length(s.points) > 2 ? 
    open_spline_path(s.points, s.v0, s.v1) :
    open_polygonal_path(s.points)
shape_path(s::ClosedSpline) = closed_spline_path(s.points)
shape_path(s::SurfaceCircle) = circular_path(s.center, s.radius)
shape_path(s::SurfaceRectangle) = rectangular_path(s.corner, s.dx, s.dy)
shape_path(s::SurfacePolygon) = closed_polygonal_path(s.vertices)
shape_path(s::SurfaceRegularPolygon) = closed_polygonal_path(regular_polygon_vertices(s.edges, s.center, s.radius, s.angle, s.inscribed))
shape_path(s::Surface) = length(s.frontier) == 1 ? shape_path(s.frontier[1]) : ClosedPathSequence([shape_path(e) for e in s.frontier])

## shape_region takes a 2D shape and computes an equivalent region
shape_region(s::Shape2D) = region(shape_path(s))

#####################################################################
## Conversions
convert(::Type{Path}, s::Shape0D) =
  and_delete_shape(shape_path(s), s)

convert(::Type{Path}, s::Shape1D) =
  and_delete_shape(shape_path(s), s)

convert(::Type{Region}, s::Shape2D) =
  and_delete_shape(shape_region(s), s)


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
prism(bs::Locs, h::Real, m=default_material) =
  prism(bs, vz(h), m)

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

@defshape(Shape1D, extruded_point, profile::Union{PointPath,Shape0D}=point_path(), v::Vec=vz(1), cb::Loc=u0())
@defshape(Shape2D, extruded_curve, profile::Union{Path,Shape1D}=circular_path(), v::Vec=vz(1), cb::Loc=u0())
@defshape(Shape3D, extruded_surface, profile::Union{Region,Shape2D}=circular_path(), v::Vec=vz(1), cb::Loc=u0())

extrusion(profile, h::Real) = extrusion(profile, vz(h))
extrusion(profile, v::Vec) =
  if profile isa PointPath || is_point(profile)
    extruded_point(profile, v)
  elseif profile isa Path || is_curve(profile)
    extruded_curve(profile, v)
  elseif profile isa Region || is_surface(profile)
    extruded_surface(profile, v)
  else
    error("Profile is neither a point nor a curve nor a surface")
  end

@defshape(Shape1D, swept_point, path::Union{Path,Shape1D}=circular_path(), profile::Union{Loc,Shape0D}=point(), rotation::Real=0, scale::Real=1)
@defshape(Shape2D, swept_curve, path::Union{Path,Shape1D}=circular_path(), profile::Union{Path,Shape1D}=circular_path(), rotation::Real=0, scale::Real=1)
@defshape(Shape3D, swept_surface, path::Union{Path,Shape1D}=circular_path(), profile::Union{Region,Shape2D}=surface_circle(), rotation::Real=0, scale::Real=1)

sweep(path, profile, rotation=0, scale=1) =
  if profile isa PointPath || is_point(profile)
    swept_point(path, profile, rotation, scale)
  elseif profile isa Path || is_curve(profile)
    swept_curve(path, profile, rotation, scale)
  elseif profile isa Region || is_surface(profile)
    swept_surface(path, profile, rotation, scale)
  else
    error("Profile is neither a point nor a curve nor a surface")
  end

@defshape(Shape1D, revolved_point, profile::Union{Loc,Point}=point(), p::Loc=u0(), n::Vec=vz(1,p.cs), start_angle::Real=0, amplitude::Real=2*pi)
@defshape(Shape2D, revolved_curve, profile::Union{Shape,Path}=line(), p::Loc=u0(), n::Vec=vz(1,p.cs), start_angle::Real=0, amplitude::Real=2*pi)
@defshape(Shape3D, revolved_surface, profile::Union{Shape,Path,Region}=circle(), p::Loc=u0(), n::Vec=vz(1,p.cs), start_angle::Real=0, amplitude::Real=2*pi)
revolve(profile::Shape=point(x(1)), p::Loc=u0(), n::Vec=vz(1,p.cs), start_angle::Real=0, amplitude::Real=2*pi) =
  if profile isa PointPath || is_point(profile)
    revolved_point(profile, p, n, start_angle, amplitude)
  elseif profile isa Path || is_curve(profile)
    revolved_curve(profile, p, n, start_angle, amplitude)
  elseif profile isa Region || is_surface(profile)
    revolved_surface(profile, p, n, start_angle, amplitude)
  else
    error("Profile is neither a point nor a curve nor a surface")
  end

@defshape(Shape1D, loft_points, profiles::Shapes0D=Shape[], rails::Shapes=Shape[], ruled::Bool=false, closed::Bool=false)
@defshape(Shape2D, loft_curves, profiles::Shapes1D=Shape[], rails::Shapes=Shape[], ruled::Bool=false, closed::Bool=false)
@defshape(Shape3D, loft_surfaces, profiles::Shapes2D=Shape[], rails::Shapes=Shape[], ruled::Bool=false, closed::Bool=false)
@defshape(Shape2D, loft_curve_point, profile::Shape1D=circle(), point::Shape0D=point(z(1)))
@defshape(Shape3D, loft_surface_point, profile::Shape2D=surface_circle(), point::Shape0D=point(z(1)))

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
export b_loft_points, b_loft_curve_point, b_loft_curves, b_loft_surfaces

# Lofts
b_loft_points(b::Backend, profiles, rails, ruled, closed, mat) =
  let f = (ruled ? (closed ? b_polygon : b_line) : (closed ? b_closed_spline : b_spline))
    and_delete_shapes(f(b, map(point_position, profiles), mat),
                      vcat(profiles, rails))
	end

b_loft_curve_point(b::Backend, profile, point) =
  let p = point_position(point),
	    path = shape_path(profile),
	    ps = path_vertices(path)
	and_delete_shapes(b_ngon(b, ps, p, is_smooth_path(path), mat),
					          [profile, point])
  end

@bdef b_loft_surface_point(profile, point)

b_loft_curves(b::Backend, profiles, rails, ruled, closed, mat) =
  b_loft(b, shape_path.(profiles), closed, ! ruled, mat)

b_loft_surfaces(b::Backend, profiles, rails, ruled, closed, mat) =
  let paths = shape_path.(profiles)
    [b_surface(b, paths[begin], mat),
     b_loft(b, paths, closed, ! ruled, mat),
     b_surface(b, paths[end], mat)]
  end

@defproxy(move, Shape3D, shape::Shape=point(), v::Vec=vx())
@defproxy(scale, Shape3D, shape::Shape=point(), s::Real=1, p::Loc=u0())
@defproxy(rotate, Shape3D, shape::Shape=point(), angle::Real=0, p::Loc=u0(), v::Vec=vz(1,p.cs))
@defproxy(transform, Shape3D, shape::Shape=point(), xform::Loc=u0())
@defproxy(mirror, Shape3D, shape::Shape=point(), p::Loc=u0(), n::Vec=vx())

export union_mirror
union_mirror(shape, p, v) =
  union(shape, mirror(shape, p, v))
#####################################################################

# We can also translate some shapes
translate(s::Line, v::Vec) = line(map(p -> p+v, s.vertices))
translate(s::Polygon, v::Vec) = polygon(map(p -> p+v, s.vertices))
translate(s::Circle, v::Vec) = circle(s.center+v, s.radius)
translate(s::Text, v::Vec) = text(s.str, s.c+v, s.h)

# We can translate arrays of Shapes
translate(ss::Shapes, v::Vec) = translate.(ss, v)

# We can compute the length of shapes as long as we can convert them
curve_length(s::Shape) = curve_length(shape_path(s))

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
## Boolean operations
export union, intersection, subtraction, empty_shape, universal_shape

struct empty_shape <: Shape0D end
struct universal_shape <: Shape3D end

@defshape(Shape2D, subtracted_surfaces, source::Shape2D=surface_circle(), mask::Shape2D=surface_rectangle())
@defshape(Shape3D, subtracted_solids, source::Shape3D=sphere(), mask::Shape3D=box())
subtraction(shape::Shape, s::empty_shape) = shape
subtraction(shape::Shape{D}, mask::Shape{D}) where D = 
  if D == 2
    subtracted_surfaces(shape, mask)
  elseif D == 3
    subtracted_solids(shape, mask)
  else
    error("Incorrect dimension for boolean operation: $(D)")
  end
subtraction(shape::Shape, shapes...) = foldl(subtraction, shapes, init=shape)
subtraction(shapes::Shapes) = subtraction(shapes...)

@defshape(Shape2D, intersected_surfaces, source::Shape2D=surface_circle(), mask::Shape2D=surface_rectangle())
@defshape(Shape3D, intersected_solids, source::Shape3D=sphere(), mask::Shape3D=box())
intersection(shape::Shape, s::universal_shape) = shape
intersection(shape::Shape{D}, mask::Shape{D}) where D = 
  if D == 2
    intersected_surfaces(shape, mask)
  elseif D == 3
    intersected_solids(shape, mask)
  else
    error("Incorrect dimension for boolean operation: $(D)")
  end
intersection(shape::Shape, shapes...) = foldl(intersection, shapes, init=shape)
intersection(shapes::Shapes) = intersection(shapes...)

@defshape(Shape2D, united_surfaces, source::Shape2D=surface_circle(), mask::Shape2D=surface_rectangle())
@defshape(Shape3D, united_solids, source::Shape3D=sphere(), mask::Shape3D=box())
union(shape::Shape, s::empty_shape) = shape
union(shape::Shape{D}, mask::Shape{D}) where D =
  if D == 2
    united_surfaces(shape, mask)
  elseif D == 3
    united_solids(shape, mask)
  else
    error("Incorrect dimension for boolean operation: $(D)")
  end
union(shape::Shape, shapes...) = foldl(union, shapes, init=shape)
union(shapes::Shapes) = union(shapes...)

@defshape(Shape3D, slice, shape::Shape3D=sphere(), p::Loc=u0(), n::Vec=vz(1))

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

@defproxy(block, Shape0D, name::String="Block", shapes::Shapes = Shape[])
@defproxy(block_instance, Shape0D, block::Block=required(), loc::Loc=u0(), scale::Real=1.0)

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
# If they are not shapes (presumably, they are paths) don't do anything
and_delete_shapes(r::Any, paths) =
  r

and_mark_deleted(b::Backend, r::Any, shape) =
  begin
    mark_deleted(b, shape)
    r
  end

    #=
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
=#
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
@defcbs highlight_shapes(ss::Shapes)
@defcbs unhighlight_shapes(ss::Shapes)
@defcbs highlight_shape(s::Shape)
@defcbs unhighlight_shape(s::Shape)
b_highlight_shapes(b::Backend{K, T}, ss::Shapes) where {K, T} =
  let refs = reduce(vcat, [collect_ref(b, s) for s in ss if realized(b, s)], init=T[])
    b_highlight_refs(b, refs)
  end
b_unhighlight_shapes(b::Backend{K, T}, ss::Shapes) where {K, T} =
  let refs = reduce(vcat, [collect_ref(b, s) for s in ss if realized(b, s)], init=T[])
    b_unhighlight_refs(b, refs)
  end
b_highlight_shape(b::Backend, s::Shape) =
  b_highlight_shapes(b, [s])

b_unhighlight_shape(b::Backend, s::Shape) =
  b_unhighlight_shapes(b, [s])

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
    map(id -> maybe_existing_shape_from_ref(b, id), f(connection(b), prompt))
  end

maybe_existing_shape_from_ref(b::Backend, r) = begin
  # If we were collecting shapes, we could check the collection
  # if traceability was on, we can search for the exact shape
  for (s, fl) in KhepriBase.shape_to_file_locations
    if ref(b, s).value == r # Found it!
      return s
    end
  end
  b_shape_from_ref(b, r)
end

@defcb render_pathname(name)

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
    [cylinder(p, rcyl, px),
     cone(px, rcon, add_x(px, lcon)),
     cylinder(p, rcyl, py),
     cone(py, rcon, add_y(py, lcon)),
     cylinder(p, rcyl, pz),
     cone(pz, rcon, add_z(pz, lcon))]
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
fill(path::Path;
    material::Material=default_surface_material(),
    backend::Backend=top_backend(),
    backends::Backends=(backend,)) =
  for backend in backends
    let mat = material_ref(backend, material)
      b_fill(backend, path, mat)
    end
  end
