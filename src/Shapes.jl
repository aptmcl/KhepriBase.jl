public GenericRef,
       NativeRef,
       NativeRefs,
       void_ref,
       ensure_ref,
       map_ref,
       ref_value,
       ref_values
export Shape,
       Shapes,
       backend,
       backend_name,
       current_backend,
       has_current_backend,
       switch_to_backend,
       is_collecting_shapes,
       with_transaction,
       with_introspection,
       merge_materials
public Backend,
       mark_deleted,
       ref, ref!,
       force_realize,
       connection,
       @deffamily,
       @defproxy
export Path,
       delete_shape, delete_shapes,
       delete_all_shapes,
       set_length_unit,
       collecting_shapes,
       collected_shapes,
       surface_boundary,
       curve_domain,
       surface_domain,
       create_layer,
       current_layer,
       delete_all_shapes_in_layer,
       subpath,
       subpath_starting_at,
       subpath_ending_at,
       bounding_box,
       capture_shape, capture_shapes,
       captured_shape, captured_shapes,
       extrusion, sweep, revolve, loft

ensure_ref(b::Backend{K,T}, r::T) where {K,T} = NativeRef{K,T}(r)
ensure_ref(b::Backend{K,T}, rs::Vector{T1}) where {K,T,T1<:T} =
  length(rs) == 1 ?
    NativeRef{K,T}(rs[1]) :
    NativeRefs{K,T}(rs)
ensure_ref(b::Backend{K,T}, rs::Vector{Vector{T1}}) where {K,T,T1<:T} =
  ensure_ref(b, collect(Iterators.flatten(rs)))
ensure_ref(b::Backend{K,T}, r::GenericRef{K,T}) where {K,T} = r
ensure_ref(b::Backend{K,T}, r::Any) where {K,T} =
  LocalRef{K,T}(r)

# This is not type-stable, but it is convenient
ref_value(b::Backend{K,T}, a::Any) where {K,T} = a
ref_value(b::Backend{K,T}, r::NativeRef{K,T}) where {K,T} = r.value
ref_value(b::Backend{K,T}, r::NativeRefs{K,T}) where {K,T} = r.values
ref_value(b::Backend{K,T}, r::LocalRef{K,T}) where {K,T} = r.value
ref_value(b::Backend, s::Proxy) = ref_value(b, ref(b, s))


# currying
public marked_deleted
ref_values(b::Backend{K,T}) where {K,T} = (r::GenericRef{K,T}) -> ref_values(b, r)
# This is type-stable
ref_values(b::Backend{K,T}, r::NativeRef{K,T}) where {K,T} = T[r.value]
ref_values(b::Backend{K,T}, r::NativeRefs{K,T}) where {K,T} = r.values
ref_values(b::Backend{K,T}, r::LocalRef{K,T}) where {K,T} = T[r.value]
ref_values(b::Backend, s::Proxy) = ref_values(b, ref(b, s))
ref_values(b::Backend{K,T}, ss::Proxies) where {K,T} =
  let refs = T[]
    for s in ss
      append!(refs, ref_values(b, ref(b, s)))
    end
    refs
  end

# currying
map_ref(f::Function, b::Backend{K,T}) where {K,T} = r -> map_ref(f, b, r)

map_ref(f::Function, b::Backend{K,T}, r::NativeRef{K,T}) where {K,T} = f(r.value)
map_ref(f::Function, b::Backend{K,T}, r::NativeRefs{K,T}) where {K,T} = map(f, r.values)
map_ref(f::Function, b::Backend, s::Proxy) = map_ref(f, b, ref(b, s))

#=
A Proxy can have subtypes whose realization might need separate reference storage 
(e.g, because we might want to delete shapes but not their materials).
This means that we need to retrieve the appropriate storage for each subtype.
=#



# This is so stupid. We need call-next-method.
really_mark_deleted(b::Backend, s::Proxy) = delete!(refs_storage(b, s), s)
really_mark_deleted(b::Backend, s::Any) = nothing
mark_deleted(b::Backend, s::Proxy) = really_mark_deleted(b, s)
# We also need to propagate this to all dependencies
mark_deleted(b::Backend, ss::Array{<:Proxy}) = foreach(s->mark_deleted(b, s), ss)
mark_deleted(b::Backend, s::Any) = nothing
marked_deleted(b::Backend, s::Proxy) = !haskey(refs_storage(b, s), s)

realize(b::Backend, s::Proxy) = error("Undefined realization for $(typeof(s)) in $(b)")

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

#=
Given a backend and a proxy, the ref function ensures that the proxy exists in the backend and returns the corresponding reference.
=#
ref(b::Backend, s::Proxy) =
  let refs = refs_storage(b, s)
    haskey(refs, s) ?
      refs[s] :
      refs[s] = ensure_ref(b, realize(b, s))
  end

ref(b::Backend) =
  s::Proxy -> ref(b, s)

reset_ref(b::Backend, s::Proxy) =
  delete!(refs_storage(b, s), s)

ref!(b::Backend, s::Proxy, r) =
  let refs = refs_storage(b, s)
    refs[s] = ensure_ref(b, r)
  end

force_realize(b::Backend, s::Proxy) =
  ref!(b, s, realize(b, s))

realized(b::Backend, s::Proxy) =
  haskey(refs_storage(b, s), s)


#=
Shapes are realized on the current_backends(), but this parameter might be
changed during realization, particularly, in a multi-threaded environment.
To avoid the problem, we capture the current value and only realize on it.
=#

force_realize(s::Proxy) = 
  let backends = current_backends()
    for b in backends
      force_realize(b, s)
    end
  end

# We can also use a shape as a surrogate for another shape

ensure_ref(b::Backend{K,T}, v::Proxy) where {K,T} =
  ref(b, v)

show(io::IO, s::Shape) =
  print(io, "$(typeof(s))(...)")

#=
Whenever a shape is created, it might be eagerly realized in its backend,
depending on the kind of shape and kind of backend (and/or its current state).
Another possibility is for the shape to be saved in some container.
It might also be necessary to record the control flow that caused the shape to be created.
This means that we need to control what happens immediately after a shape is initialized.
The protocol after_init takes care of that.
=#

after_init(a::Any) = a
after_init(s::Proxy) =
  begin
    maybe_realize(s)
    s
  end
# UniqueProxy is ... unique! We need to ensure that there is only one instance of it, at least, in what concerns its specification.

equal_spec(s1, s2) = s1 == s2
equal_spec(s1::BackendParameter, s2::BackendParameter) =
  s1 === s2 || (s1.default === s2.default && isequal(pairs(s1.value), pairs(s2.value)))

after_init(s::UniqueProxy) =
  let bs = current_backends(),
      unique(s) = begin
        for b in bs
          for (si, _) in refs_storage(b, s)
            if equal_spec(s, si)
              return si
            end
          end
        end
        s
      end,
      unique_s = unique(s)
    maybe_realize(unique_s)
    unique_s
  end

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
with_transaction(fn) =
  with_transaction(fn, current_backends())

with_transaction(fn, bs::Backends) =
  isempty(bs) ?
    fn() :
    with_transaction(() -> with_transaction(fn, Base.tail(bs)), first(bs))

maybe_realize(s) =
  for b in current_backends()
    try
      maybe_realize(b, s)
    catch e
      handle_backend_error(e, b)
    end
  end

#=
Backends might need to immediately realize a shape while supporting further modifications
e.g., using boolean operations. Others, however, cannot do that and can only realize
shapes when they have complete information about them.
We are going to implement these two strategies using transactions. The first strategy
corresponds to an auto-commit transaction, while the second corresponds to a manual commit transaction.
=#
public Transaction, AutoCommitTransaction, ManualCommitTransaction
abstract type Transaction end
struct AutoCommitTransaction <: Transaction end
struct ManualCommitTransaction <: Transaction
  proxies::Vector{<:Proxy}
  ManualCommitTransaction() = new(Proxy[])
end
#=
When a backend is in auto-commit mode, proxies are immediately realized.
When a backend is in manual-commit mode, proxies are collected and only realized when the transaction ends.
=#
# By default, backends have a field called transaction containing a Parameter to a transaction.
current_transaction(b) = b.transaction

start_transaction(b) = start_transaction(b, current_transaction(b)())
commit_transaction(b) = commit_transaction(b, current_transaction(b)())
abort_transaction(b) = abort_transaction(b, current_transaction(b)())

start_transaction(b, t::AutoCommitTransaction) = error("Cannot start an auto-commit transaction")
commit_transaction(b, t::AutoCommitTransaction) = error("Cannot commit an auto-commit transaction")
abort_transaction(b, t::AutoCommitTransaction) = error("Cannot abort an auto-commit transaction")

start_transaction(b, t::ManualCommitTransaction) =
  isempty(t.proxies) ? nothing : error("Cannot start a new transaction in backend $b while another one is active")

commit_transaction(b, t::ManualCommitTransaction) =
  while !isempty(t.proxies)
    let s = popfirst!(t.proxies)
      if ! realized(b, s)
        force_realize(b, s)
      end
    end
  end
abort_transaction(b, t::ManualCommitTransaction) =
  empty!(t.proxies)

maybe_realize(b, s) = maybe_realize(current_transaction(b)(), b, s)
maybe_delete(b, s) = maybe_delete(current_transaction(b)(), b, s)

maybe_realize(::AutoCommitTransaction, b, s) =
  if ! realized(b, s)
    force_realize(b, s)
  end
maybe_delete(::AutoCommitTransaction, b, s) =
  if realized(b, s)
    b_delete_refs(b, ref_values(b, s))
    reset_ref(b, s)
  end

maybe_realize(t::ManualCommitTransaction, b, s) =
  push!(t.proxies, s)
maybe_delete(t::ManualCommitTransaction, b, s) =
  if realized(b, s)
    b_delete_refs(b, ref_values(b, s))
    reset_ref(b, s)
  else
    let idx = findfirst(==(s), t.proxies)
      if isnothing(idx)
        @warn("Shape $s is not realized in $b neither is present in the current transaction")
      else
        deleteat!(t.proxies, idx)
      end
    end
  end

struct IntrospectionTransaction <: Transaction end
maybe_realize(::IntrospectionTransaction, b, s) = nothing
maybe_delete(::IntrospectionTransaction, b, s) = nothing

with_introspection(f, b) =
  with(current_transaction(b), IntrospectionTransaction()) do
    f()
  end

ensure_manual_commit_transaction(b) =
  let t = current_transaction(b)()
    t isa ManualCommitTransaction ?
      t :
      ManualCommitTransaction()
  end

with_transaction(f, b) =
  with(current_transaction(b), ensure_manual_commit_transaction(b)) do
    start_transaction(b)
    try
      f()
      commit_transaction(b)
    catch e
      abort_transaction(b)
      rethrow(e)
    end
  end

#=
Frequently, we need to collect all shapes that are created:
=#

is_collecting_shapes = Parameter(false)
collected_shapes = Parameter(Shape[])
collect_shape!(s::Shape) = (push!(collected_shapes(), s); s)
collecting_shapes(fn) =
    with(collected_shapes, Shape[]) do
        with(is_collecting_shapes, true) do
            fn()
        end
        collected_shapes()
    end
maybe_collect(s::Shape) = (is_collecting_shapes() && collect_shape!(s); s)

######################################################
#Traceability
traceability = Parameter(false)
trace_depth = Parameter(1000)
excluded_modules = Parameter([Base, Base.CoreLogging, KhepriBase])
# We use a dict from shapes to file locations
# and a dict from file locations to shapes
shape_to_file_locations = IdDict()
file_location_to_shapes = Dict()
const trace_lock = ReentrantLock()

public traceability, trace_depth,
       excluded_modules, clear_trace!,
       shape_source, source_shapes,
       shape_to_file_locations, file_location_to_shapes,
       highlight_source_shapes

shape_source(s) =
  lock(trace_lock) do
    copy(get(shape_to_file_locations, s, []))
  end
source_shapes(file::Symbol, line) =
  lock(trace_lock) do
    copy(get(file_location_to_shapes, (file, line), Shape[]))
  end
source_shapes(file::String, line) = source_shapes(Symbol(file), line)
traced_shapes() =
  lock(trace_lock) do
    collect(keys(shape_to_file_locations))
  end
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
  lock(trace_lock) do
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
    lock(trace_lock) do
      shape_to_file_locations[s] = locations
      for location in locations
        push!(get!(file_location_to_shapes, location, Shape[]), s)
      end
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

#=
We have a hierarchy of deletion.
delete_all_annotations() # Deletes shape annotations but not the underlying shapes.
delete_all_shapes()      # Deletes all shapes but not the underlying resources (e.g., materials)
delete_all()             # Deletes all shapes and the underlying resources (e.g., materials, layers, etc)
=#

@defcbs delete_all_annotations()
@defcbs delete_all_shapes()
@defcbs delete_all()

#=
Deleting one annotation can, as a side effect, delete others it depends on, so a
later shape in the iteration may already be gone. The previous code expressed this
with an empty `catch`, which silently swallowed *every* exception — not just the
benign already-deleted case — hiding genuine failures (a retired/transport-broken
backend, a malformed ref, a bug in delete_shape) while `empty!(refs)` below still
wiped the annotation dict, orphaning the backend's real objects.

Two deliberate choices avoid that:
  - We snapshot `keys(refs)` before iterating, because delete_shape mutates this
    very dict (via reset_ref/really_mark_deleted) and mutating a Dict mid-iteration
    is undefined in Julia. The snapshot also makes the already-deleted skip meaningful.
  - `marked_deleted(b, shape)` makes the "already deleted" intent explicit and is
    exception-free (delete_shape on a gone shape is a no-op via maybe_delete's
    `realized` guard, not an error), so any exception that *does* escape delete_shape
    is a real failure and is routed to handle_backend_error, which retires a dead
    RemoteBackend on the transport-error set and rethrows everything else — the same
    policy used by maybe_realize and the @defcbs/@defcb frontend wrappers.
See also: marked_deleted, maybe_delete, handle_backend_error.
=#
b_delete_all_annotations(b::Backend) =
  let refs = annotation_refs_storage(b)
    for shape in collect(keys(refs))
      marked_deleted(b, shape) && continue
      try
        delete_shape(shape)
      catch e
        handle_backend_error(e, b)
      end
    end
    empty!(refs)
  end

b_delete_all_shapes(b::Backend) =
  let refs = shape_refs_storage(b)
    clear_trace!()
    b_delete_all_annotations(b)
    b_delete_all_shape_refs(b)
    empty!(refs)
  end


@defcb all_shapes()
b_all_shapes(b::T) where {T<:Backend} =
  b_all_shapes(shape_storage_type(T), b)

# Local: shapes are dict keys — O(n), no reconstruction
b_all_shapes(::LocalShapeStorage, b) =
  b_created_shapes(b)

# Remote: reconstruct Shape wrappers from remote refs
b_all_shapes(::RemoteShapeStorage, b) =
  b_existing_shapes(b)

@bdef(b_shape_from_ref(r))
@bdef(b_create_shape_from_ref_value(r))

export created_shapes, existing_shapes

created_shapes(; backend::Backend=top_backend()) =
  b_created_shapes(backend)

existing_shapes(; backend::Backend=top_backend()) =
  b_existing_shapes(backend)

@defcbs set_length_unit(unit::String="")
@defcb reset_backend()
@defcb save_as(pathname::String, format::String)

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
  abstract_name = esc(Symbol("Abstract", typename))
  field_names = map(field -> field.args[1].args[1], fields)
  field_types = map(field -> esc(field.args[1].args[2]), fields)
  field_inits = map(field -> field.args[2], fields)
#  field_renames = map(esc ∘ Symbol ∘ uppercasefirst ∘ string, field_names)
  field_renames = map(Symbol ∘ string, field_names)
  field_replacements = Dict(zip(field_names, field_renames))
  struct_fields = map((name,typ) -> :($(name) :: $(typ)), field_names, field_types) # To ensure identity, we need a mutable struct with const fields.
  # It seems Julia 1.9 does not support const fields.
  #struct_fields = map((name,typ) -> Expr(:const, Expr(:(::), name, typ)), field_names, field_types)
  mk_param(name,typ,init) = Expr(:kw, name, init) #Expr(:kw, Expr(:(::), name, typ), init)
  opt_params = map(mk_param, field_renames, field_types, map(init -> replace_in(init, field_replacements), field_inits))
  key_params = map(mk_param, field_names, field_types, field_renames)
  constructor_name = esc(name)
  predicate_name = esc(Symbol("is_", name_str))
  mk_convert(name,typ) = :($(esc(name)))
  field_converts = map(mk_convert, field_names, field_types)
  selector_names = map(field_name -> esc(Symbol(name_str, "_", string(field_name))), field_names)
  # Generate docstring from field metadata
  field_type_strs = map(field -> string(field.args[1].args[2]), fields)
  field_init_strs = map(field -> string(field.args[2]), fields)
  sig_parts = join(["$(fn)=$(fi)" for (fn, fi) in zip(field_names, field_init_strs)], ", ")
  field_lines = ["- `$(fn)::$(ft)` — default: `$(fi)`"
                 for (fn, ft, fi) in zip(field_names, field_type_strs, field_init_strs)]
  src_file = string(__source__.file)
  src_line = __source__.line
  docstr = string("    ", name_str, "(", sig_parts, ")\n\n",
                  "Create a `", typename, "` (`", parent, "`).\n\n",
                  "Defined at `", src_file, ":", src_line, "`\n\n",
                  "# Fields\n",
                  join(field_lines, "\n"), "\n\n",
                  "Backend operation: `b_", name_str, "`\n")
  quote
    export $(constructor_name), $(struct_name), $(predicate_name), $(selector_names...)
    $(Expr(:public, abstract_name))
    abstract type $(abstract_name) <: $(parent) end
    mutable struct $struct_name <: $(abstract_name)
      $(struct_fields...)
    end
    # we don't need to convert anything because Julia already does that with the default constructor
    # and, by the same idea, we don't need to define parameter types.
    @noinline $(constructor_name)($(opt_params...); $(key_params...)) =
      after_init($(struct_name)($(field_converts...)))
    $(predicate_name)(v::$(struct_name)) = true
    $(predicate_name)(v::Any) = false
    $(map((selector_name, field_name) -> :($(selector_name)(v::$(struct_name)) = v.$(field_name)),
          selector_names, field_names)...)
    $(__source__)
    KhepriBase.realize(b::Backend, s::$(abstract_name)) =
      $(Symbol(:b_, name))(b, $(map(f->:(ref_value(b, getproperty(s, $(QuoteNode(f))))), field_names)...))
    KhepriBase.mark_deleted(b::Backend, v::$(abstract_name)) =
      if ! marked_deleted(b, v)
        really_mark_deleted(b, v)
        $(map(field_name -> :(mark_deleted(b, v.$(field_name))), field_names)...)
      end
    KhepriBase.meta_program(v::$(abstract_name)) =
        Expr(:call, $(Expr(:quote, name)), $(map(field_name -> :(meta_program(v.$(field_name))), field_names)...))
    KhepriBase.equal_spec(s1::$(abstract_name), s2::$(abstract_name)) =
      $(reduce((a,b) -> :($a && $b),
               map(field_name -> :(equal_spec(s1.$(field_name), s2.$(field_name))), field_names),
               init=:true))
    Base.@doc $(docstr) $(constructor_name)
  end
end

#=
There are entities who have parameters that depend on the type of backend, e.g., AutoCAD, Rhino, Revit, etc.
As a concrete example, a material might have widely different realizations, depending on the backend. This
is similar to what happens with shapes. The difference is that we want to change the way a material is
instantiated in a backend without requiring additional generic function specializations.
=#

set_on!(tb::Type{<:Backend}, proxy, ref) =
  begin
    proxy.data(tb, ref)
    for b in current_backend()
      if b isa tb
        reset_ref(b, proxy)
      end
    end
    proxy
  end

#=
Layers are just a classification mechanism.
Some backends, however, can colorize the shapes that have that layer, or can make
those shapes visible or invisible.
=#
@defproxy(base_layer, Layer)
@defproxy((layer, StandardLayer), Layer, name::String="Layer", visible::Bool=true, color::RGBA=rgba(1,1,1,1))
export layer
create_layer(args...; kargs...) =
  let s = layer(args...; kargs...)
    force_realize(s)
    s
  end
#=
realize(b::Backend, l::StandardLayer) =
  b_layer(b, l.name, l.visible, l.color)
=#
current_layer(backends::Backends=current_backends()) =
  isempty(backends) ?
    layer("Default") :
    let b1 = first(backends),
        layer = get_or_create_layer_from_ref_value(b1, b_current_layer_ref(b1))
      for b in Base.tail(backends)
        ref!(b, layer, b_current_layer_ref(b))
      end
      layer
    end
@bdef(b_create_layer_from_ref_value(r))

current_layer(layer, backends::Backends=current_backends()) =
  for b in backends
    b_current_layer_ref(b, ref_value(b, layer))
  end
delete_all_shapes_in_layer(layer, backends::Backends=current_backends()) =
  for b in backends
    b_delete_all_shapes_in_layer(b, ref_value(b, layer))
  end

#=
Materials
A shape can be directly associated to a material or the shape can be associated
to a layer and the layer is then associated to the material
=#
@defproxy((layer_material, MaterialInLayer), Material, layer::Layer=current_layer(), data::BackendParameter=BackendParameter())
layer_material(name::String, bvs...) = layer_material(layer(name, true), BackendParameter(bvs...))
# Some backends prefer to use layers instead of materials
export use_layers_for_materials, with_material_as_layer
const use_layers_for_materials = Parameter(false)
with_material_as_layer(f::Function, b::Backend, m::Material) =
  use_layers_for_materials() ?
    let cur_layer = b_current_layer_ref(b),
        new_layer = ref_value(b, layer(m))
      cur_layer == new_layer ?
        f() :
        begin
          b_current_layer_ref(b, new_layer)
          let res = f()
            b_current_layer_ref(b, cur_layer)
            res
          end
        end
      end :
    f()

# The ref_value of a BackendParameter is the ref_value for parameter value for the backend
ref_value(b::Backend{K,T}, data::BackendParameter) where {K,T} = ref_value(b, data(typeof(b)))

b_layer_material(b::Backend, data) =
  b_get_material(b, data)

# By default, the layer is ignored in the creation of materials, as only a few backends depend on that.
b_layer_material(b::Backend, layer, spec) =
  b_get_material(b, spec)

#=
Material references are stored on each backend. They might be erased when needed (but that should be the exception, not the rule).
=#

#delete_all_materials()

public merge_backend_materials
merge_materials(materials...) =
  let name = join([material_name(material) for material in materials], "_"),
      newdata = IdDict{Backend,Any}()
    for material in reverse(materials)
      # Violating an abstraction barrier!
      for (b, v) in material.data.value
        newdata[b] = haskey(newdata, b) ? merge_backend_materials(b, v, newdata[b]) : v
      end
    end
    layer_material(name, newdata...)
  end
merge_backend_materials(b::Backend, v1, v2) = v1

# For compatibility
export set_material
const set_material = set_on!
# To facilitate accessing the material reference that is provided to the backends:
material_ref(b::Backend, m::Material) = ref_value(b, m)
material_ref(b::Backend, s::Shape) = material_ref(b, s.material)
material_ref(b::Backend, s::Annotation) = material_ref(b, s.material)

#=
A material follows Filament's standard PBR model.
See Materials.jl for the full Filament property documentation.
The realize method calls b_material, which cascades through
decreasing parameter tiers until a backend-supported level is reached.
=#
@defproxy(pbr_material, Material, name::String="Material",
  base_color::RGBA=rgba(0.5, 0.5, 0.5, 1.0),
  metallic::Real=0.0,
  roughness::Real=0.5,
  specular::Real=0.35,
  sheen_color::RGB=rgb(0.0, 0.0, 0.0),
  sheen_roughness::Real=0.0,
  clearcoat::Real=0.0,
  clearcoat_roughness::Real=0.0,
  anisotropy::Real=0.0,
  anisotropy_direction::RGB=rgb(0.0, 0.0, 0.0),
  ambient_occlusion::Real=0.0,
  normal::RGB=rgb(0.0, 0.0, 0.0),
  bent_normal::RGB=rgb(0.0, 0.0, 0.0),
  clearcoat_normal::RGB=rgb(0.0, 0.0, 0.0),
  emission_color::RGBA=rgba(0.0, 0.0, 0.0, 0.0),
  emission_strength::Real=1.0,
  post_lighting_color::RGBA=rgba(0.0, 0.0, 0.0, 0.0),
  ior::Real=1.5,
  transmission::Real=0.0,
  transmission_roughness::Real=0.5,
  absorption::Real=0.0,
  micro_thickness::Real=0.0,
  thickness::Real=0.0,
  data::BackendParameter=BackendParameter())

# Override auto-generated meta_program for PbrMaterial to use keyword
# arguments and skip the internal BackendParameter data field.
meta_program(m::PbrMaterial) =
  # A canonical material round-trips as its bare binding symbol; anything else as a full literal.
  get(_canonical_material_symbols, m, nothing) !== nothing ?
    _canonical_material_symbols[m] :
  let defaults = (base_color=rgba(0.5, 0.5, 0.5, 1.0),
                  metallic=0.0, roughness=0.5, specular=0.35,
                  sheen_color=rgb(0,0,0), sheen_roughness=0.0,
                  clearcoat=0.0, clearcoat_roughness=0.0,
                  anisotropy=0.0, anisotropy_direction=rgb(0,0,0),
                  ambient_occlusion=0.0, normal=rgb(0,0,0),
                  bent_normal=rgb(0,0,0), clearcoat_normal=rgb(0,0,0),
                  emission_color=rgba(0,0,0,0), emission_strength=1.0,
                  post_lighting_color=rgba(0,0,0,0),
                  ior=1.5, transmission=0.0, transmission_roughness=0.5,
                  absorption=0.0, micro_thickness=0.0, thickness=0.0),
      fields = [:base_color, :metallic, :roughness, :specular,
                :sheen_color, :sheen_roughness,
                :clearcoat, :clearcoat_roughness,
                :anisotropy, :anisotropy_direction,
                :ambient_occlusion, :normal, :bent_normal, :clearcoat_normal,
                :emission_color, :emission_strength, :post_lighting_color,
                :ior, :transmission, :transmission_roughness,
                :absorption, :micro_thickness, :thickness],
      kwargs = [Expr(:kw, f, meta_program(getfield(m, f)))
                for f in fields
                if getfield(m, f) != defaults[f]]
    Expr(:call, :pbr_material,
      Expr(:parameters, kwargs...),
      meta_program(m.name))
  end

# When a PbrMaterial has a backend-specific override (set via set_material),
# use the override via b_get_material. Otherwise, use b_material which cascades
# through decreasing parameter tiers until a backend-supported level is reached.
realize(b::Backend, s::PbrMaterial) =
  let spec = s.data(typeof(b))
    spec !== nothing ?
      b_get_material(b, spec) :
      b_material(b, s.name, s.base_color, s.metallic, s.roughness,
                 s.specular,
                 s.ior, s.transmission, s.transmission_roughness,
                 s.clearcoat, s.clearcoat_roughness,
                 s.emission_color, s.emission_strength,
                 s.sheen_color, s.sheen_roughness,
                 s.anisotropy, s.anisotropy_direction,
                 s.ambient_occlusion, s.normal, s.bent_normal, s.clearcoat_normal,
                 s.post_lighting_color,
                 s.absorption, s.micro_thickness, s.thickness)
  end

# material() is the unified constructor:
#   material(base_color=red)             → PbrMaterial (via pbr_material)
#   material("Glass", base_color=…)      → PbrMaterial (via pbr_material)
#   material(my_layer)                   → MaterialInLayer (via layer_material)
#   material("name", Backend=>spec, …)   → MaterialInLayer (via layer_material, for backend overrides)
export material, standard_material
# Deprecated material keyword aliases (fields renamed for cross-backend consistency,
# and emission_strength promoted). Kept for one release: translate old names to the
# canonical ones with a one-time warning. (Direct pbr_material callers should migrate.)
const material_kwarg_aliases = Dict(
  :reflectance => :specular,
  :clear_coat => :clearcoat,
  :clear_coat_roughness => :clearcoat_roughness,
  :clear_coat_normal => :clearcoat_normal,
  :emissive => :emission_color)
migrate_material_kwargs(kwargs) =
  let out = Pair{Symbol,Any}[]
    for (k, v) in kwargs
      if haskey(material_kwarg_aliases, k)
        nk = material_kwarg_aliases[k]
        @warn "Khepri: material keyword `$k` is deprecated; use `$nk`." maxlog=1
        push!(out, nk => v)
      else
        push!(out, k => v)
      end
    end
    out
  end
material(; kwargs...) = pbr_material(; migrate_material_kwargs(kwargs)...)
material(name::String; kwargs...) = pbr_material(name; migrate_material_kwargs(kwargs)...)
material(l::Layer, data::BackendParameter=BackendParameter()) = layer_material(l, data)
material(name::String, bvs::Pair...) = layer_material(name, bvs...)
const standard_material = pbr_material

# Pre-defined materials with physically-based properties.
# These provide sensible PBR defaults so that backends implementing
# b_material get meaningful visual results out of the box.
export material_point, material_curve, material_surface,
       material_basic, material_glass,
       material_metal, material_wood,
       material_concrete, material_plaster,
       material_grass, material_clay

const material_point = material(
  name="Points",
  base_color=rgba(1.0, 1.0, 0.0, 1.0),
  roughness=1.0,
  data=BackendParameter(default=backend_default))
const material_curve = material(
  name="Curves",
  base_color=rgba(0.0, 0.0, 0.0, 1.0),
  roughness=1.0,
  data=BackendParameter(default=backend_default))
const material_surface = material(
  name="Surfaces",
  base_color=rgba(0.8, 0.8, 0.8, 1.0),
  roughness=0.5,
  data=BackendParameter(default=backend_default))
#=
Canonical architectural materials.

Why these specific PBR values. Each material's parameters are tuned so that
the default realization path (`realize(backend, PbrMaterial)` →
`b_material(backend, name, base_color, metallic, roughness, ...)`) produces
visually-comparable output across every production backend — the same
script should look like the same building regardless of whether Mitsuba,
Blender Cycles, Rhino, AutoCAD, or POVJay is rendering it.

  - Metals: `roughness=0.15` is tight enough to show clear reflections of the
    sky/environment while avoiding a mirror-finish look; `metallic=1.0`
    directs Principled-BSDF-family backends to use the base colour as F0.
  - Glass: `roughness=0.0` keeps the surface optically clear; `transmission=
    0.95` leaves a hint of Fresnel reflection. The base-colour alpha stays
    opaque (1.0) so non-transmission backends (TikZ, SVG) still draw a
    visible surface.
  - Diffuse materials (wood, concrete, plaster, grass, clay): `specular`
    values are chosen to give a plausible Fresnel F0 under daylight; concrete
    has a slightly lower specular than plaster, which in turn is lower
    than clay, matching real-world observations.

See also: `architectural_materials` in ArchMaterials.jl (the iterable
version used by cross-backend tests and by the default-material closures in
KhepriBlender / KhepriMitsuba / …).
=#
const material_basic = material(
  name="Basic",
  base_color=rgba(0.70, 0.70, 0.70, 1.0),
  roughness=0.5,
  specular=0.5,
  data=BackendParameter(default=backend_default))
const material_glass = material(
  name="Glass",
  base_color=rgba(0.95, 0.97, 1.0, 1.0),
  roughness=0.0,
  specular=0.5,
  ior=1.5,
  transmission=0.95)
const material_metal = material(
  name="Metal",
  base_color=rgba(0.80, 0.80, 0.82, 1.0),
  metallic=1.0,
  roughness=0.15,
  specular=0.5)
const material_wood = material(
  name="Wood",
  base_color=rgba(0.50, 0.32, 0.18, 1.0),
  roughness=0.7,
  specular=0.5)
const material_concrete = material(
  name="Concrete",
  base_color=rgba(0.65, 0.63, 0.60, 1.0),
  roughness=0.9,
  specular=0.4)
const material_plaster = material(
  name="Plaster",
  base_color=rgba(0.92, 0.91, 0.88, 1.0),
  roughness=0.8,
  specular=0.3)
const material_grass = material(
  name="Grass",
  base_color=rgba(0.30, 0.50, 0.20, 1.0),
  roughness=0.95,
  specular=0.2)
const material_clay = material(
  name="Clay",
  base_color=rgba(0.85, 0.78, 0.70, 1.0),
  roughness=0.9,
  specular=0.4)

# Registry mapping each canonical material to its binding symbol, so an introspected model emits a bare
# `material_concrete` rather than a verbose `pbr_material("Concrete"; …)` literal. `meta_program(::Material)`
# consults this before falling back to the full-literal emit.
const _canonical_material_symbols = IdDict{Material,Symbol}(
  material_point => :material_point, material_curve => :material_curve,
  material_surface => :material_surface, material_basic => :material_basic,
  material_glass => :material_glass, material_metal => :material_metal,
  material_wood => :material_wood, material_concrete => :material_concrete,
  material_plaster => :material_plaster, material_grass => :material_grass,
  material_clay => :material_clay)

# Polymorphic material accessors — uniform interface for all Material subtypes
export material_color, material_name

# Color representative of a material's appearance
material_color(m::PbrMaterial) = m.base_color
material_color(m::MaterialInLayer) = m.layer.color
material_color(::Material) = rgba(0.5, 0.5, 0.5, 1.0)

# Name of a material
material_name(m::PbrMaterial) = m.name
material_name(m::MaterialInLayer) = m.layer.name
material_name(::Material) = "Material"

# Derive a layer from a material (for use_layers_for_materials mechanism).
# Extends layer() via dispatch: layer(material) → the layer representing that material.
layer(m::PbrMaterial) = layer(m.name, true, m.base_color)
layer(m::MaterialInLayer) = m.layer
layer(::Material) = layer("Default")

export default_point_material, default_curve_material, default_surface_material, default_material
const default_point_material = Parameter{Material}(material_point)
const default_curve_material = Parameter{Material}(material_curve)
const default_surface_material = Parameter{Material}(material_surface)
const default_material = Parameter{Material}(material_basic)

# To handle the direct use of shapes as paths:
b_stroke(b::Backend, path::Shape, mat) =
  and_mark_deleted(b, ref_value(b, path), path)


# This might be usable, so
public @defproxy, realize, realized, Shape0D, Shape1D, Shape2D, Shape3D, void_ref

#=
A shape is just a proxy with a set of configurable defaults (e.g., material, layer, etc) and a 
standard method specialization for the realize function.
=#
macro defshape(supertype, name_typename, fields...)
  (name, typename) = name_typename isa Symbol ?
    (name_typename, Symbol(string(map(uppercasefirst,split(string(name_typename),'_'))...))) :
    name_typename.args
  field_names = map(field -> field.args[1].args[1], fields)
  n_fields = length(field_names)
  b_func = Symbol(:b_, name)
  default_material =
    supertype == :Shape0D ? :default_point_material :
    supertype == :Shape1D ? :default_curve_material :
    supertype == :Shape2D ? :default_surface_material :
    supertype == :Shape3D ? :default_material :
    supertype == :Annotation ? :default_annotation_material :
    error("Unknown supertype:", supertype)
  src = __source__
  src_file_str = string(src.file)
  src_line = src.line
  esc(quote
    @defproxy($(name_typename), $(supertype), $(fields...), material::Material=$(default_material)())
    $(src)
    KhepriBase.realize(b::Backend, s::$(typename)) =
      $(b_func)(b, $(map(f->:(getproperty(s, $(QuoteNode(f)))), field_names)...), material_ref(b, s))
    # Arity verification: b_<name> must accept (backend, fields..., material)
    let b_sym = $(QuoteNode(b_func))
      if isdefined($(__module__), b_sym)
        let expected = Tuple{Backend, $(fill(:Any, n_fields + 1)...)}
          hasmethod($(b_func), expected) ||
            @warn string("@defshape ", $(string(name)),
                         " at ", $(src_file_str), ":", $(src_line),
                         ": ", $(string(b_func)),
                         " has no method accepting ", $(n_fields + 2),
                         " args (backend + ", $(n_fields), " fields + material)")
        end
      end
    end
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

findkey_that(p, dict) = begin
  for k in keys(dict)
    if p(k)
      return k
    end
  end
  nothing
end

prev_annotation_that(p, b=top_backend()) =
  let dict = annotation_refs_storage(b),
      ann = findkey_that(p, dict)
    isnothing(ann) ?
      nothing :
      begin
         delete_shape(ann) # We delete it so that we can replace it with a new one
         ann
      end
  end

@defshape(Annotation, dimension, from::Loc=u0(), to::Loc=ux(), text::Any=string(distance(from, to)), size::Real=1, offset::Real=0.1)
@defshape(Annotation, arc_dimension, center::Loc=u0(), radius::Real=1, start_angle::Real=0, amplitude::Real=pi, radius_text::Any=string(radius), amplitude_text::Any=string(amplitude), size::Real=1, offset::Real=0.1)

@defshape(Annotation, labels, p::Loc=u0(), data::Vector{@NamedTuple{txt::Any, mat::Material, scale::Real}}=[(txt="Hello, World!", mat=default_annotation_material(), scale=1.0)])

export default_annotation_material
const default_annotation_material = Parameter{Material}(material(layer("annotation", true, rgba(0, 0, 0.5, 1.0))))
export default_annotation_scale
const default_annotation_scale = Parameter{Real}(1.0)

existing_material(mat, mats) =
  any(m -> material_color(m) == material_color(mat), mats)

equal_illustration_properties(i1, i2) =
  i1.txt == i2.txt &&
  material_color(i1.mat) == material_color(i2.mat) &&
  i1.scale == i2.scale

export label
label(p, txt, mat=default_annotation_material(), scale=default_annotation_scale()) =
  let ann = prev_annotation_that(ann->is_labels(ann) && isapprox(p, ann.p, atol=1e-3)),
      new_d = (txt=txt, mat=mat, scale=scale);
    isnothing(ann) ?
      labels(p, [new_d]) :
      any(d->equal_illustration_properties(new_d, d), ann.data) ?
        labels(p, ann.data) : 
        labels(p, [ann.data..., new_d])
  end

@defshape(Annotation, vectors_illustration, start::Loc=u0(), angle::Real=0, 
  radii::Vector{Real}=[],
  radii_texts::Vector{Any}=[],
  mats::Vector{Material}=[])
export vector_illustration
vector_illustration(p, a, r, txt, mat=default_annotation_material()) =
  let ann = prev_annotation_that(ann->is_vectors_illustration(ann) && isequal(p, ann.start) && isequal(a, ann.angle))
    isnothing(ann) ?
      vectors_illustration(p, a, [r], [txt], [mat]) :
      (txt, material_color(mat)) in zip(ann.radii_texts, [material_color(m) for m in ann.mats]) ?
        vectors_illustration(p, a, ann.radii, ann.radii_texts, ann.mats) :
        vectors_illustration(p, a, [ann.radii..., r], [ann.radii_texts..., txt], [ann.mats..., mat])
  end

@defshape(Annotation, radii_illustration, center::Loc=u0(),
  radii::Vector{Real}=[],
  radii_texts::Vector{Any}=[],
  mats::Vector{Material}=[])
export radius_illustration
radius_illustration(c, r, txt, mat=default_annotation_material()) =
  let ann = prev_annotation_that(ann->is_radii_illustration(ann) && isequal(c, ann.center))
    isnothing(ann) ?
      radii_illustration(c, [r], [txt], [mat]) :
      (txt, material_color(mat)) in zip(ann.radii_texts, [material_color(m) for m in ann.mats]) ?
        radii_illustration(c, ann.radii, ann.radii_texts, ann.mats) :
        radii_illustration(c, [ann.radii..., r], [ann.radii_texts..., txt], [ann.mats..., mat])
  end

@defshape(Annotation, angles_illustration, center::Loc=u0(), 
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
    isnothing(ann) ?
      angles_illustration(c, [r], [s], [a], [r_txt], [s_txt], [a_txt], [mat]) :
      angles_illustration(c, [ann.radii..., r], 
                           [ann.start_angles..., s],
                           [ann.amplitudes..., a],
                           [ann.radii_texts..., r_txt], 
                               [ann.start_angles_texts..., s_txt], 
                               [ann.amplitudes_texts..., a_txt],
                               [ann.mats..., mat])
  end

# This is similar to an angles illustration but not entirely equal
@defshape(Annotation, arcs_illustration, center::Loc=u0(),
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
    isnothing(ann) ?
      arcs_illustration(c, [r], [s], [a], [r_txt], [s_txt], [a_txt], [mat]) :
      arcs_illustration(c, [ann.radii..., r],
                           [ann.start_angles..., s],
                           [ann.amplitudes..., a],
                           [ann.radii_texts..., r_txt],
                           [ann.start_angles_texts..., s_txt], 
                           [ann.amplitudes_texts..., a_txt], 
                           [ann.mats..., mat])
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
@defshape(Shape2D, surface_ring, center::Loc=u0(), inner_radius::Real=0.5, outer_radius::Real=1)
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

#=
A surface_path is the filled region bounded by a single closed path. Unlike the
bare @defproxy it used to be, it is now a @defshape so it carries a `material`
field like every other Shape2D surface (surface_circle, surface_polygon, ...);
without it, the same colored-surface script could fill a surface_polygon but not
a surface_path, breaking backend portability.

@defshape generates `realize(b,s) = b_surface_path(b, s.path, material_ref(b,s))`,
so the backend operation b_surface_path must exist. It is provided once, backend-
agnostically, in Backend.jl as `b_surface_path(b, path, mat) = b_surface(b, path,
mat)`: a ClosedPath is already a valid b_surface input, so no per-backend method
is needed. The previous hand-written `realize` (commented out) routed through the
legacy backend_fill path, which throws UndefinedBackendException in modern b_*
backends; it is intentionally dropped.
See also: b_surface (Backend.jl), surface (this file).
=#
@defshape(Shape2D, surface_path, path::ClosedPath=[circular_path()])
surface_boundary(s::Shape2D, backend::Backend=top_backend()) =
  b_surface_boundary(backend, s)
public b_surface_boundary
b_surface_boundary(b::Backend, s::Shape2D) = throw(UndefinedBackendException())

curve_domain(s::Shape1D, backend::Backend=top_backend()) =
  b_curve_domain(backend, s)
public b_curve_domain
b_curve_domain(b::Backend, s::Shape1D) = throw(UndefinedBackendException())
# THIS SHOULD APPLY ONLY TO Shape1D but was generalized to Shape to apply to Move
map_division(f::Function, s::Shape1D, n::Int) =
  map_division(f, shape_path(s), n)

surface_domain(s::Shape2D, backend::Backend=top_backend()) =
  b_surface_domain(backend, s)
public b_surface_domain
b_surface_domain(b::Backend, s::Shape2D) = throw(UndefinedBackendException())
map_division(f::Function, s::Shape2D, nu::Int, nv::Int) =
  b_map_division(backend(s), f, s, nu, nv)

# b_map_division must be `public` so @import_backend_api pulls it into backend
# scope: the per-backend specializations (KhepriRhino, KhepriAutoCAD) are written
# bare (`b_map_division(b::RH, ...)`) and only extend THIS function — rather than
# defining a private one — when the name is imported.
public b_map_division
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
export shape_path
shape_path(s::Point) = point_path(s.position)
shape_path(s::Circle) = circular_path(s.center, s.radius)
shape_path(s::Arc) = arc_path(s.center, s.radius, s.start_angle, s.amplitude)
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
shape_path(s::RegularPolygon) =
  closed_polygonal_path(regular_polygon_vertices(s.edges, s.center, s.radius, s.angle, s.inscribed))
shape_path(s::SurfaceCircle) = circular_path(s.center, s.radius)
shape_path(s::SurfaceRectangle) = rectangular_path(s.corner, s.dx, s.dy)
shape_path(s::SurfacePolygon) = closed_polygonal_path(s.vertices)
shape_path(s::SurfaceRegularPolygon) = closed_polygonal_path(regular_polygon_vertices(s.edges, s.center, s.radius, s.angle, s.inscribed))
shape_path(s::SurfaceRing) = circular_path(s.center, s.outer_radius)
shape_path(s::Surface) = length(s.frontier) == 1 ? shape_path(s.frontier[1]) : closed_path_sequence(shape_path.(s.frontier)...)

## shape_region takes a 2D shape and computes an equivalent region
shape_region(s::Shape2D) = region(shape_path(s))
#=
SurfaceRing has an inner hole that the default `region(shape_path(s))` would
silently drop, producing a solid disk on extrusion instead of an annulus.
The override carries both boundary loops into the resulting Region.
=#
shape_region(s::SurfaceRing) =
  region(circular_path(s.center, s.outer_radius),
         circular_path(s.center, s.inner_radius))

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
prism(bs::Locs, h::Real, m=default_material()) =
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
cone(cb::Loc, r::Real, ct::Loc, _material=default_material(); material=_material) =
  let (c, h) = position_and_height(cb, ct)
    cone(c, r, h, material)
  end
@defshape(Shape3D, cone_frustum, cb::Loc=u0(), rb::Real=1, h::Real=1, rt::Real=1)
cone_frustum(cb::Loc, rb::Real, ct::Loc, rt::Real, _material=default_material(); material=_material) =
  let (c, h) = position_and_height(cb, ct)
    cone_frustum(c, rb, h, rt, material)
  end
@defshape(Shape3D, cylinder, cb::Loc=u0(), r::Real=1, h::Real=1)
cylinder(cb::Loc, r::Real, ct::Loc, _material=default_material(); material=_material) =
  let (c, h) = position_and_height(cb, ct)
    cylinder(c, r, h, material)
  end

# cs-based solids store their base center in a local coordinate system (the axis is the cs's +z), but
# meta_program(::Loc) emits a Loc's LOCAL coordinates and drops the cs — so a positioned/tilted cylinder,
# cone or cone_frustum would round-trip to the origin, axis-aligned. Emit the two world endpoints via the
# 2-point constructor (as meta_program(::Beam) does) so both position and orientation survive.
meta_program(s::Cylinder) =
  Expr(:call, :cylinder, meta_program(in_world(s.cb)), meta_program(s.r),
       meta_program(in_world(add_z(s.cb, s.h))), meta_program(s.material))
meta_program(s::Cone) =
  Expr(:call, :cone, meta_program(in_world(s.cb)), meta_program(s.r),
       meta_program(in_world(add_z(s.cb, s.h))), meta_program(s.material))
meta_program(s::ConeFrustum) =
  Expr(:call, :cone_frustum, meta_program(in_world(s.cb)), meta_program(s.rb),
       meta_program(in_world(add_z(s.cb, s.h))), meta_program(s.rt), meta_program(s.material))

# regular_pyramid/prism/pyramid_frustum are cb-based with NO 2-point constructor, so preserve the base
# loc's FULL coordinate system (position + orientation) via loc_from_o_vx_vy — the auto-generated
# meta_program routes cb through meta_program(::Loc) (local coords, cs dropped), collapsing a tilted solid
# to the origin, axis-aligned.
_loc_with_cs_expr(cb) =
  let o = in_world(cb)
    Expr(:call, :loc_from_o_vx_vy, meta_program(o),
         meta_program(in_world(add_x(cb, 1.0)) - o),
         meta_program(in_world(add_y(cb, 1.0)) - o))
  end
meta_program(s::RegularPyramid) =
  Expr(:call, :regular_pyramid, meta_program(s.edges), _loc_with_cs_expr(s.cb),
       meta_program(s.rb), meta_program(s.angle), meta_program(s.h),
       meta_program(s.inscribed), meta_program(s.material))
meta_program(s::RegularPrism) =
  Expr(:call, :regular_prism, meta_program(s.edges), _loc_with_cs_expr(s.cb),
       meta_program(s.r), meta_program(s.angle), meta_program(s.h),
       meta_program(s.inscribed), meta_program(s.material))
meta_program(s::RegularPyramidFrustum) =
  Expr(:call, :regular_pyramid_frustum, meta_program(s.edges), _loc_with_cs_expr(s.cb),
       meta_program(s.rb), meta_program(s.angle), meta_program(s.h),
       meta_program(s.rt), meta_program(s.inscribed), meta_program(s.material))

#=
Transformation proxies wrap an existing shape with a deferred transformation.
They SHOULD be type-parametric — moving a Shape3D should yield a Shape3D —
but `@defproxy` cannot parameterize the declared supertype on a field, so all
five wrappers lie and declare `Shape1D` regardless of what they wrap. Until
that parametric redesign, the `shape_dim` trait (Types.jl) — not the type
parameter — carries the truth: the Union method below recurses into the
wrapped shape (handling nesting such as `move(rotate(...))`), and all
dimension classification (`is_curve`/`is_surface`/`is_solid`, hence the
`extrusion`/`sweep`/`revolve`/CSG dispatchers) must go through it instead of
dispatching on `Shape{D}`. Wrapper fields are `shape::Shape`, so Paths and
Regions cannot be wrapped; CSG results (e.g. UnitedSolids <: Shape3D)
classify via the parametric `shape_dim` default. Known follow-up gaps:
`loft` of transformed surfaces now reaches `loft_surfaces`, whose `Shapes2D`
field still rejects wrappers, and `extruded_point` does not accept wrapped
points.

See also: shape_dim (Types.jl); the extruded_surface/swept_surface profile
fields below, which accept TransformedShape.
=#
@defproxy(move, Shape1D, shape::Shape=point(), v::Vec=vx())
@defproxy(scale, Shape1D, shape::Shape=point(), s::Real=1, p::Loc=u0())
@defproxy(rotate, Shape1D, shape::Shape=point(), angle::Real=0, p::Loc=u0(), v::Vec=vz(1,p.cs))
@defproxy(transform, Shape1D, shape::Shape=point(), xform::Loc=u0())
@defproxy(mirror, Shape1D, shape::Shape=point(), p::Loc=u0(), n::Vec=vx())

"Union of the five transformation proxies; their dimension comes from `shape_dim`."
const TransformedShape = Union{Move,Rotate,Scale,Mirror,Transform}

shape_dim(s::TransformedShape) = shape_dim(s.shape)

export union_mirror
union_mirror(shape, p, v) =
  union(shape, mirror(shape, p, v))

#=
The profile fields below are deliberately Union types instead of plain Path /
Region. With a strictly-typed field, the @defproxy constructor would call
`convert(Path, profile)` (resp. `convert(Region, profile)`), which deletes the
input shape and rebuilds an equivalent Path/Region via `shape_path`/
`shape_region`. That conversion loses native references and forces the default
fallback in Backend.jl to use surface_grid sampling — which (a) drops the
backend-native sweep/extrude/revolve and (b) breaks for Shape2D inputs that
have no `shape_path` method (e.g. UnitedSurfaces from `union(surface_circles)`
in map30-04b/`pilar_circulos_radiais`).

Backends that can build natively from a shape proxy override
`b_extruded_*(::Shape*, ...)` (see KhepriAutoCAD/src/AutoCAD.jl line 832 and
KhepriRhino/src/Rhino.jl line 579) and walk the existing refs via `map_ref`.
Backends without such overrides still see Path/Region inputs through the
default fallback chain.
=#
#=
TransformedShape joins the surface-profile Unions (here and in swept_surface)
because the wrappers are declared Shape1D regardless of what they wrap: a
Move-wrapped surface routed here by the `extrusion`/`sweep` dispatchers —
which classify via `is_surface`/`shape_dim`, not via the field type — would
otherwise be rejected at construction. Same precedent as the CSG operand
fields (loosened to `Shape` for the wrappers' sake); the dimension check is
the dispatcher's job.
=#
@defshape(Shape1D, extruded_point, profile::Union{PointPath,Shape0D}=point_path(), v::Vec=vz(1), cb::Loc=u0())
@defshape(Shape2D, extruded_curve, profile::Union{Path,Shape1D}=circular_path(), v::Vec=vz(1), cb::Loc=u0())
@defshape(Shape3D, extruded_surface, profile::Union{Region,Shape2D,TransformedShape}=region(circular_path()), v::Vec=vz(1), cb::Loc=u0())

extrusion(profile, h::Real, args...; kargs...) = extrusion(profile, vz(h), args...; kargs...)
extrusion(profile, v::Vec, args...; kargs...) =
  if profile isa PointPath || is_point(profile)
    extruded_point(profile, v, args...; kargs...)
  elseif profile isa Path || is_curve(profile)
    extruded_curve(profile, v, args...; kargs...)
  elseif profile isa Region || is_surface(profile)
    extruded_surface(profile, v, args...; kargs...)
  else
    error("Profile is neither a point nor a curve nor a surface")
  end

# Sweep mirrors extrusion: Union types let proxies retain Shape inputs so that
# backend overrides keyed on `Shape1D`/`Shape2D` (AutoCAD line 870, Rhino line
# 619) reach the native plugin sweep without re-realizing through path frames.
@defshape(Shape1D, swept_point, path::Union{Path,Shape1D}=circular_path(), profile::Union{Path,Shape0D}=point_path(), rotation::Real=0, scale::Real=1)
@defshape(Shape2D, swept_curve, path::Union{Path,Shape1D}=circular_path(), profile::Union{Path,Shape1D}=circular_path(), rotation::Real=0, scale::Real=1)
@defshape(Shape3D, swept_surface, path::Union{Path,Shape1D}=circular_path(), profile::Union{Region,Shape2D,TransformedShape}=region(circular_path()), rotation::Real=0, scale::Real=1)

sweep(path, profile, args...; kargs...) =
  if profile isa PointPath || is_point(profile)
    swept_point(path, profile, args...; kargs...)
  elseif profile isa Path || is_curve(profile)
    swept_curve(path, profile, args...; kargs...)
  elseif profile isa Region || is_surface(profile)
    swept_surface(path, profile, args...; kargs...)
  else
    error("Profile is neither a point nor a curve nor a surface")
  end

#=
The Union-typed profile fields above keep the input Shape alive until realize
time so that native backends can map_ref its refs. On storage backends
(LocalBackend: TikZ, POVRay, ...) that broke the construction-time contract
the strict Path/Region fields used to provide via `convert` in two ways:
(a) the profile shape stayed in the realize queue and realized *standalone*,
leaking its bytes into the output ahead of its consumer, and (b) the
realize-time `convert` → `and_delete_shape` mutated `local_shape_storage`
while `realize_shapes` iterated it, shifting indices and silently skipping
the shapes that followed (predioSinusoidal lost all four extruded solids and
six cylinders this way). Consuming the inputs here, at proxy construction,
restores the pre-Union semantics for storage backends without giving up the
native-backend win: `b_consume_shape` is a deliberate no-op everywhere except
LocalBackend, so native refs stay alive for map_ref.

Scope: exactly the six proxies whose fields were widened to Unions. The
`surface()` frontier, `revolve`, and `loft` inputs keep their blessed
realize-time deletion (their boundary curves are expected to appear in the
output); the snapshot in `realize_shapes` (Backends.jl) defends against
their mid-pass mutation.

See also: consume_shape!/b_consume_shape (near b_delete_shape, below);
realize_shapes and b_consume_shape(::LocalBackend) in Backends.jl.
=#
after_init(s::Union{AbstractExtrudedPoint,AbstractExtrudedCurve,AbstractExtrudedSurface}) =
  begin
    consume_shape!(s.profile)
    Base.@invoke after_init(s::Shape)
  end
after_init(s::Union{AbstractSweptPoint,AbstractSweptCurve,AbstractSweptSurface}) =
  begin
    consume_shape!(s.path)
    consume_shape!(s.profile)
    Base.@invoke after_init(s::Shape)
  end

@defshape(Shape1D, revolved_point, profile::Union{Loc,Point}=point(), p::Loc=u0(), n::Vec=vz(1,p.cs), start_angle::Real=0, amplitude::Real=2*pi)
@defshape(Shape2D, revolved_curve, profile::Union{Shape,Path}=line(), p::Loc=u0(), n::Vec=vz(1,p.cs), start_angle::Real=0, amplitude::Real=2*pi)
@defshape(Shape3D, revolved_surface, profile::Union{Shape,Path,Region}=region(circular_path()), p::Loc=u0(), n::Vec=vz(1,p.cs), start_angle::Real=0, amplitude::Real=2*pi)
# Profile is untyped so Path and Region can also be passed alongside Shape;
# the body checks them explicitly. `extrusion` and `sweep` follow the same
# pattern.
revolve(profile=point(x(1)), args...; kargs...) =
  if profile isa PointPath || (profile isa Shape && is_point(profile))
    revolved_point(profile, args...; kargs...)
  elseif profile isa Path || (profile isa Shape && is_curve(profile))
    revolved_curve(profile, args...; kargs...)
  elseif profile isa Region || (profile isa Shape && is_surface(profile))
    revolved_surface(profile, args...; kargs...)
  else
    error("Profile is neither a point nor a curve nor a surface")
  end

@defshape(Shape1D, loft_points, profiles::Shapes0D=Shape0D[], rails::Shapes=Shape[], ruled::Bool=false, closed::Bool=false)
@defshape(Shape2D, loft_curves, profiles::Shapes1D=Shape1D[], rails::Shapes=Shape[], ruled::Bool=false, closed::Bool=false)
@defshape(Shape3D, loft_surfaces, profiles::Shapes2D=Shape2D[], rails::Shapes=Shape[], ruled::Bool=false, closed::Bool=false)
@defshape(Shape2D, loft_curve_point, profile::Shape1D=circle(), point::Shape0D=point(z(1)))
@defshape(Shape3D, loft_surface_point, profile::Shape2D=surface_circle(), point::Shape0D=point(z(1)))

loft(profiles::Shapes=Shape[], args...; kargs...) =
  #= A loft interpolates a solid/surface/curve BETWEEN cross sections, so it
     needs at least two of them. The dispatch below keys on `all(is_point,
     profiles)`, `all(is_curve, profiles)`, etc., and `all` is VACUOUSLY TRUE
     on an empty vector: an empty `profiles` therefore silently routes to
     `loft_points`, which either throws an opaque `MethodError` (when the
     element type cannot convert to `Shape0D`, e.g. `loft(Shape[])`) or
     silently builds a degenerate zero-point spline (`loft(Shape0D[])`). A
     single profile is likewise geometrically meaningless. Guard up front so
     callers get an actionable message instead of a downstream MethodError or
     silent garbage. `loft_ruled` delegates here, so it is covered too. =#
  if length(profiles) < 2
    error("loft requires at least two cross sections; got $(length(profiles))")
  elseif all(is_point, profiles)
    loft_points(profiles, args...; kargs...)
  elseif all(is_curve, profiles)
    loft_curves(profiles, args...; kargs...)
  elseif all(is_surface, profiles)
    loft_surfaces(profiles, args...; kargs...)
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
public b_loft_points, b_loft_curve_point, b_loft_curves, b_loft_surfaces

# Lofts
b_loft_points(b::Backend, profiles, rails, ruled, closed, mat) =
  let f = (ruled ? (closed ? b_polygon : b_line) : (closed ? b_closed_spline : b_spline))
    and_delete_shapes(f(b, map(point_position, profiles), mat),
                      vcat(profiles, rails))
	end

# Default loft from a curve to a point: emit an n-gon with apex at the
# point. Earlier signature lacked `mat` and the body referenced an
# undefined `mat` symbol — broken for any backend that didn't override.
b_loft_curve_point(b::Backend, profile, point, mat) =
  let p = point_position(point),
      path = shape_path(profile),
      ps = path_vertices(path)
    and_delete_shapes(b_ngon(b, ps, p, is_smooth_path(path), mat),
                      [profile, point])
  end

# Default surface→point loft delegates to curve→point on the surface's
# boundary. Backends with a native solid loft (AutoCAD, Rhino) override.
b_loft_surface_point(b::Backend, profile, point, mat) =
  b_loft_curve_point(b, profile, point, mat)

b_loft_curves(b::Backend, profiles, rails, ruled, closed, mat) =
  and_delete_shapes(
    b_loft(b, shape_path.(profiles), closed, ! ruled, mat),
    profiles)

b_loft_surfaces(b::Backend{K,T}, profiles, rails, ruled, closed, mat) where {K,T} =
  let paths = shape_path.(profiles),
      top = b_surface(b, paths[begin], mat),
      sides = b_loft(b, paths, closed, ! ruled, mat),
      bottom = b_surface(b, paths[end], mat)
    b_solidify(b, T[top, (sides isa Vector ? sides : [sides])..., bottom])
  end

#####################################################################

shape_path(s::Move) = translate(shape_path(s.shape), s.v)
shape_region(s::Move) = translate(shape_region(s.shape), s.v)

#=
Realize-time plumbing for transformed surface profiles. The widened
`extruded_surface`/`swept_surface` profile fields accept TransformedShape, so
the default backend fallbacks must be able to lower a wrapper to a Region.
These are the TransformedShape companions to Backend.jl's
`b_extruded_surface(::Backend, ::Shape2D, ...)` and
`b_swept_surface(::Backend, path, ::Shape2D, ...)`; they live here (not in
Backend.jl) because Backend.jl is included before this file, where
TransformedShape is defined. Only Move has a `shape_region`; the other
wrappers error loudly until their lowering is implemented (follow-up,
together with the parametric wrapper redesign).
=#
convert(::Type{Region}, s::TransformedShape) =
  and_delete_shape(shape_region(s), s)
shape_region(s::Union{Rotate,Scale,Mirror,Transform}) =
  error("shape_region is not yet implemented for $(typeof(s)); only Move-wrapped surfaces can be lowered to a Region")
b_extruded_surface(b::Backend, profile::TransformedShape, v, cb, bmat, tmat, smat) =
  b_extruded_surface(b, convert(Region, profile), v, cb, bmat, tmat, smat)
b_swept_surface(b::Backend, path, profile::TransformedShape, rotation, scaling, mat) =
  b_swept_surface(b, path, convert(Region, profile), rotation, scaling, mat)


# We can also translate some shapes
translate(s::Line, v::Vec) = line(map(p -> p+v, s.vertices))
translate(s::Polygon, v::Vec) = polygon(map(p -> p+v, s.vertices))
translate(s::Circle, v::Vec) = circle(s.center+v, s.radius)
translate(s::Text, v::Vec) = text(s.str, s.corner+v, s.height)

# We can translate arrays of Shapes
translate(ss::Shapes, v::Vec) = translate.(ss, v)

# We can compute the length of shapes as long as we can convert them
curve_length(s::Shape) = curve_length(shape_path(s))

# We will also need to compute a bounding rectangle
bounding_rectangle(s::Union{Line, Polygon}) =
    bounding_rectangle(s.vertices)

bounding_rectangle(pts::Locs) =
    #= A bounding rectangle is seeded from pts[1]; with no points there is no
    corner to seed min/max from. Fail loudly with an actionable message rather
    than throwing a raw BoundsError on pts[1] (e.g. when an empty Line/Polygon
    or an empty selection flows in through bounding_rectangle(ss::Shapes)). =#
    isempty(pts) ?
        error("Cannot compute a bounding rectangle of an empty set of points") :
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

#=
Boolean source/mask types are declared `Shape` (not `Shape{D}`) so that
transformation wrappers — Move, Rotate, Scale, Mirror, Transform — can pass
through. Those wrappers are declared as `Shape1D` in `@defproxy` regardless
of the wrapped shape's dimension; without the loosening, dispatch on
`Shape{D}` rejects e.g. `subtracted_solids(::Box, ::Rotate{Cylinder})`.
The runtime dimension is recovered via the `shape_dim` trait (Types.jl).
=#
@defshape(Shape2D, subtracted_surfaces, source::Shape=surface_circle(), mask::Shape=surface_rectangle())
@defshape(Shape3D, subtracted_solids, source::Shape=sphere(), mask::Shape=box())
@defshape(Shape2D, intersected_surfaces, source::Shape=surface_circle(), mask::Shape=surface_rectangle())
@defshape(Shape3D, intersected_solids, source::Shape=sphere(), mask::Shape=box())
@defshape(Shape2D, united_surfaces, source::Shape=surface_circle(), mask::Shape=surface_rectangle())
@defshape(Shape3D, united_solids, source::Shape=sphere(), mask::Shape=box())

#=
Path of a CSG surface proxy. UnitedSurfaces / SubtractedSurfaces /
IntersectedSurfaces represent CSG operations on 2D surfaces and have no
single boundary path: a union of two disjoint discs is a 2D area with two
outer loops; a union of overlapping discs is one region with a merged outer;
a subtraction can produce regions with holes or carve a region into multiple
pieces; an intersection can produce nothing or many pieces. None of these
maps cleanly to a Khepri `Path` (a 1-D closed loop) or a `Region` (one outer
+ holes).

We return a `PathSet` of the operand boundaries — semantically "a set of
independent paths." This is correct when the resulting topology is "list of
disjoint outer boundaries" (the common case for `union(surface_circle,
surface_circle, ...)` of non-overlapping circles, e.g. map30-04b
`pilar_circulos_radiais`). It is **incorrect** when operands overlap or when
the operation introduces holes — `convert(ClosedPath, ::PathSet)` then
collapses the set via `subtract_paths`, treating the first as outer and the
rest as holes (Paths.jl line 1115), which silently produces wrong geometry.

A correct fix would require a `MultiRegion` type (a vector of Regions, each
its own outer + holes) and downstream support in `b_extruded_surface`,
`b_swept_surface`, and friends. Until then, backends with a native CSG
operation (AutoCAD, Rhino) override `b_united_surfaces` to walk the operand
refs directly via `map_ref` and bypass this conversion entirely (see
`KhepriAutoCAD/src/AutoCAD.jl` `b_united_surfaces(::ACAD)`); this
shape_path fallback is only exercised by backends that have neither a
native CSG nor a Shape2D-aware extrude (Blender's mesh-based pipeline).
=#
#=
Recursive `shape_path` on nested CSG proxies (e.g. `union(c0, c1, ..., c5)`
expands to `UnitedSurfaces(UnitedSurfaces(...), c5)` 5 levels deep) produces
nested PathSets, but `convert(ClosedPath, ::PathSet)` only knows how to fold
over flat path lists — it raises `Cannot convert PathSet to
ClosedPolygonalPath` when an inner element is itself a PathSet. We flatten
to a single-level path_set so the foldl-via-subtract_paths conversion can
proceed.
=#
flatten_paths(p::PathSet) = mapreduce(flatten_paths, vcat, p.paths; init=Path[])
flatten_paths(p::Path) = Path[p]

unsupported_surface_csg_path(op) =
  throw(ArgumentError("Cannot derive a single boundary path for a 2D surface $(op) on this backend. Use a backend with native 2D CSG support or convert the result to explicit regions before realization."))

shape_path(s::UnitedSurfaces) = unsupported_surface_csg_path("union")
shape_path(s::SubtractedSurfaces) = unsupported_surface_csg_path("subtraction")
shape_path(s::IntersectedSurfaces) = unsupported_surface_csg_path("intersection")

subtraction(shape::Shape, s::empty_shape) = shape
subtraction(shape::Shape, mask::Shape) =
  shape_dim(shape) == shape_dim(mask) ?
    (shape_dim(shape) == 2 ? subtracted_surfaces(shape, mask) :
     shape_dim(shape) == 3 ? subtracted_solids(shape, mask) :
     error("subtraction not defined for $(shape_dim(shape))D shapes")) :
    error("Cannot subtract: dimensions differ ($(shape_dim(shape))D vs $(shape_dim(mask))D)")
subtraction(shape::Shape, mask1::Shape, mask2::Shape, masks...) =
  foldl(subtraction, (mask1, mask2, masks...), init=shape)
subtraction(shapes::Shapes) = subtraction(shapes...)

intersection(shape::Shape, s::universal_shape) = shape
intersection(shape::Shape, mask::Shape) =
  shape_dim(shape) == shape_dim(mask) ?
    (shape_dim(shape) == 2 ? intersected_surfaces(shape, mask) :
     shape_dim(shape) == 3 ? intersected_solids(shape, mask) :
     error("intersection not defined for $(shape_dim(shape))D shapes")) :
    error("Cannot intersect: dimensions differ ($(shape_dim(shape))D vs $(shape_dim(mask))D)")
intersection(shape::Shape, mask1::Shape, mask2::Shape, masks...) =
  foldl(intersection, (mask1, mask2, masks...), init=shape)
intersection(shapes::Shapes) = intersection(shapes...)

union(shape::Shape, s::empty_shape) = shape
union(shape::Shape, mask::Shape) =
  shape_dim(shape) == shape_dim(mask) ?
    (shape_dim(shape) == 2 ? united_surfaces(shape, mask) :
     shape_dim(shape) == 3 ? united_solids(shape, mask) :
     error("union not defined for $(shape_dim(shape))D shapes")) :
    error("Cannot union: dimensions differ ($(shape_dim(shape))D vs $(shape_dim(mask))D)")
union(shape::Shape, mask1::Shape, mask2::Shape, masks...) =
  foldl(union, (mask1, mask2, masks...), init=shape)
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

public b_frame_at
b_frame_at(b::Backend, s::Shape2D, u::Real, v::Real) =
  missing_specialization(b, :b_frame_at, s, u, v)

frame_at(c::Shape1D, t::Real) = b_frame_at(backend(c), c, t)
frame_at(s::Shape2D, u::Real, v::Real) = b_frame_at(backend(s), s, u, v)

#Some specific cases can be handled in an uniform way without the backend
frame_at(s::SurfaceRectangle, u::Real, v::Real) = add_xy(s.corner, u, v)
frame_at(s::SurfaceCircle, u::Real, v::Real) = add_pol(s.center, u, v)

@defshape(Shape2D, surface_grid, points::Matrix{<:Loc}=zeros(Loc,(2,2)),
          closed_u::Bool=false, closed_v::Bool=false,
          smooth_u::Bool=true, smooth_v::Bool=true)

surface_grid(_points::Vector{<:Vector{<:Loc}},
  _closed_u=false, _closed_v=false, _smooth_u=true, _smooth_v=true, _material=default_material();
  points=_points, closed_u=_closed_u, closed_v=_closed_v, smooth_u=_smooth_u, smooth_v=_smooth_v, material=_material) =
  surface_grid(stack(_points, dims=1), closed_u, closed_v, smooth_u, smooth_v, material=material)
# For interpolator to work, we need this:

convert(::Type{Matrix{XYZ}}, ptss::Vector{Vector{<:Loc}}) =
  stack(ptss, dims=1)

surface_domain(s::SurfaceGrid) = (0.0, 1.0, 0.0, 1.0)
frame_at(s::SurfaceGrid, u::Real, v::Real) = evaluate(s, u, v)
#=
No `backend` parameter here, unlike `surface_domain`/`surface_boundary` above.
A SurfaceGrid is an analytically-known surface: `frame_at(s, u, v)` resolves to
`evaluate(s, u, v)`, a local interpolation over the grid's stored control points,
so sampling needs no backend round-trip. The earlier signature carried an unused
`backend::Backend=top_backend()` that was silently dropped — a caller passing a
backend would have seen it ignored. Dropping it keeps this method consistent with
its more general sibling `map_division(f, s::Shape2D, nu, nv)`, which likewise
takes no explicit backend and derives one via `backend(s)` when one is needed.
See also: frame_at(s::SurfaceGrid, ...), surface_domain(s::SurfaceGrid).
=#
map_division(f::Function, s::SurfaceGrid, nu::Int, nv::Int) =
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

# Groups (reusable collections of shapes, instantiated at multiple locations)
# The factory function, when provided, allows backends without native group
# support to re-create member shapes at each instance location.

@defproxy(group, Shape0D, name::String="Group", shapes::Shapes = Shape[], factory::Union{Function,Nothing} = nothing)
@defproxy(group_instance, Shape0D, group::Group=required(), loc::Loc=u0())

# Default group realization for backends without a native group concept: the group itself is just a
# container, and each instance re-creates the member shapes (via the factory) at the instance
# location. Backends with native groups (Revit's CreateGroup) override both methods.
realize(b::Backend, s::Group) = void_ref(b)
realize(b::Backend, s::GroupInstance) =
  with(current_cs, translated_cs(current_cs(), cx(s.loc), cy(s.loc), cz(s.loc))) do
    let factory = s.group.factory
      factory === nothing ? void_ref(b) :
        (ref_values(b, collecting_shapes(factory)); void_ref(b))
    end
  end

################################################################################
## Key locations for bounding box approximation

export shape_locs
shape_locs(s::Shape) = Loc[]

# 1D shapes
shape_locs(s::Line) = s.vertices
shape_locs(s::Polygon) = s.vertices
shape_locs(s::ClosedLine) = s.vertices
shape_locs(s::Spline) = s.points
shape_locs(s::ClosedSpline) = s.points
shape_locs(s::Circle) =
  let c = s.center, r = s.radius
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r)]
  end
#=
`shape_locs` feeds the conservative bounding-box approximation in
`shapes_bbox` (Backend.jl): the box is the AABB of the returned points,
so every point of the real shape must be enclosed by them.

For an Arc we deliberately return the four cardinal extents of the FULL
circle (center ± radius along x and y) rather than the true extrema of the
swept sub-arc derived from `start_angle`/`amplitude` (both in radians). This
is an intentional, safe over-approximation: the full-circle box always
contains the arc, so bounds stay correct; it merely makes them looser (e.g.
a 30° arc still reports the whole circle's box). We keep it because the tight
extrema (the two endpoints plus every cardinal axis crossed within the
sweep) cost extra trig per query for a 1D curve that rarely dominates a
scene's bounds, and because some backends realize an Arc through a native op
that keeps this proxy while others fall back to a SurfacePolygon whose own
`shape_locs` already yields the tighter box — both are valid framings.
Switch to true extrema only if a use-case needs tight arc bounds and can
absorb the reframing of any auto-fit camera.
See also: shape_locs(::Circle), shapes_bbox, surface_arc stress test.
=#
"Key bbox locations for an Arc: the full circle's cardinal extents — a conservative over-approximation that ignores start_angle/amplitude (radians) on purpose."
shape_locs(s::Arc) =
  let c = s.center, r = s.radius
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r)]
  end
shape_locs(s::Rectangle) =
  [s.corner, add_xy(s.corner, s.dx, s.dy)]
shape_locs(s::RegularPolygon) =
  let c = s.center, r = s.radius
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r)]
  end

# 2D shapes (surfaces)
shape_locs(s::SurfacePolygon) = s.vertices
shape_locs(s::SurfaceCircle) =
  let c = s.center, r = s.radius
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r)]
  end
shape_locs(s::SurfaceRectangle) =
  [s.corner, add_xy(s.corner, s.dx, s.dy)]
shape_locs(s::SurfaceGrid) = vec(s.points)
shape_locs(s::SurfaceRegularPolygon) =
  let c = s.center, r = s.radius
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r)]
  end

# 3D shapes
shape_locs(s::Sphere) =
  let c = s.center, r = s.radius
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r),
     add_z(c, r), add_z(c, -r)]
  end
shape_locs(s::Box) =
  [s.c, add_xyz(s.c, s.dx, s.dy, s.dz)]
shape_locs(s::Cylinder) =
  let c = s.cb, r = s.r
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r),
     add_xz(c, r, s.h), add_xz(c, -r, s.h)]
  end
shape_locs(s::Cone) =
  let c = s.cb, r = s.r
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r), add_z(c, s.h)]
  end
shape_locs(s::ConeFrustum) =
  let c = s.cb, r = max(s.rb, s.rt)
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r), add_z(c, s.h)]
  end
shape_locs(s::Torus) =
  let c = s.center, r = s.re + s.ri
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r),
     add_z(c, s.ri), add_z(c, -s.ri)]
  end
shape_locs(s::RegularPyramid) =
  let c = s.cb, r = s.rb
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r), add_z(c, s.h)]
  end
shape_locs(s::RegularPyramidFrustum) =
  let c = s.cb, r = max(s.rb, s.rt)
    [add_x(c, r), add_x(c, -r), add_y(c, r), add_y(c, -r), add_z(c, s.h)]
  end

# Compound shapes
shape_locs(s::ExtrudedSurface) =
  let base_locs = try path_vertices(outer_path(s.profile)) catch; Loc[] end
    vcat(base_locs, [p + s.v for p in base_locs])
  end
shape_locs(s::ExtrudedCurve) =
  let base_locs = try path_vertices(s.profile) catch; Loc[] end
    vcat(base_locs, [p + s.v for p in base_locs])
  end
shape_locs(s::SweptCurve) =
  try path_vertices(s.path) catch; Loc[] end
shape_locs(s::SweptSurface) =
  try path_vertices(s.path) catch; Loc[] end

################################################################################
bounding_box(shape::Shape) =
  bounding_box([shape])

bounding_box(shapes::Shapes=Shape[]) =
  if isempty(shapes)
    [u0(), u0()]
  else
    b_bounding_box(backend(shapes[1]), shapes)
  end

public b_bounding_box
b_bounding_box(backend::Backend, shape::Shape) =
  throw(UndefinedBackendException())

@defcbs delete_shape(s::Proxy)

b_delete_shape(b::Backend, s::Proxy) =
  maybe_delete(b, s)

#=
Construction-time consumption of extrusion/sweep profile inputs (see the
after_init overrides for the extruded/swept proxies). The generic default is
a *deliberate* no-op, not an unfinished stub — same contract as
b_set_layer_material (Backends.jl): the whole point of the Union-typed
profile fields is that on native backends the input shape's refs stay alive
so b_extruded_*/b_swept_* overrides can map_ref them at realize time;
destroying or even unqueuing anything here would undo that. Only backends
that *queue* shapes for a later realize pass override this (LocalBackend, in
Backends.jl) to unqueue the profile so it never realizes standalone —
unqueue, not destroy: that override filter!s local_shape_storage and the
layer index but never touches refs.

See also: b_delete_shape (above), b_consume_shape(::LocalBackend) and
realize_shapes in Backends.jl.
=#
"Backend hook: a profile shape was consumed at construction by a containing proxy."
b_consume_shape(b::Backend, s::Shape) = nothing

#=
Frontend dispatch for profile consumption. Path/Region (and Loc/PointPath)
inputs fall through the ::Any no-op — there is nothing queued to unqueue.
TransformedShape wrappers recurse: both the wrapper and the shape it wraps
sit in the realize queue, and the realize-time lowering
(convert(::Type{Region}, ::TransformedShape), below) consumes both.
=#
"Mark `s` as consumed by a containing proxy on all current backends."
consume_shape!(s::Shape) =
  for b in current_backends()
    b_consume_shape(b, s)
  end
consume_shape!(s::TransformedShape) =
  begin
    consume_shape!(s.shape)
    for b in current_backends()
      b_consume_shape(b, s)
    end
  end
consume_shape!(s::Any) = nothing

delete_shapes(ss::Proxies=Proxy[], bs=current_backends()) =
  for s in ss
    delete_shape(s, bs)
  end

export and_delete_shape, and_delete_shapes, and_mark_deleted
and_delete_shape(r::Any, shape::Proxy) =
  begin
    delete_shape(shape)
    r
  end

and_delete_shapes(r::Any, shapes::Proxies) =
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
  let refs = realized_ref_values(b, ss)
    b_highlight_refs(b, refs)
  end
b_unhighlight_shapes(b::Backend{K, T}, ss::Shapes) where {K, T} =
  let refs = realized_ref_values(b, ss)
    b_unhighlight_refs(b, refs)
  end
realized_ref_values(b::Backend{K, T}, ss::Shapes) where {K, T} =
  let refs = T[]
    for s in ss
      realized(b, s) && append!(refs, ref_values(b, s))
    end
    refs
  end
b_highlight_shape(b::Backend, s::Shape) =
  b_highlight_shapes(b, [s])

b_unhighlight_shape(b::Backend, s::Shape) =
  b_unhighlight_shapes(b, [s])

export unhighlight_all_shapes
const unhighlight_all_shapes = unhighlight_all_refs


capture_shape(s=select_shape("Select shape to be captured")) =
  if ! isnothing(s)
    b_generate_captured_shape(backend(s), s)
  end

capture_shapes(ss=select_shapes("Select shapes to be captured")) =
    # A cancelled/empty selection yields no shapes; nothing to capture (cf. capture_shape's isnothing guard).
    if ! isempty(ss)
        b_generate_captured_shapes(backend(ss[1]), ss)
    end

# `capture_shape` PRINTS replay code shaped like `captured_shape(<backend>, <handle>)`.
# `captured_shape`/`captured_shapes` are therefore an intentional EXCEPTION to the b_*
# hook convention: they are the user-facing names the *emitted* code calls (evaluated
# later in a plain script after `using KhepriXXX`), so they keep their bare names and
# are EXPORTED — backends specialize them via `KhepriBase.captured_shape(b::XX, ...)`.
# The generators that PRODUCE that code are ordinary internal backend ops, so they stay
# b_*: `b_generate_captured_shape`/`b_generate_captured_shapes`.
captured_shape(b::Backend, handle) = throw(UndefinedBackendException())
captured_shapes(b::Backend, handles) = throw(UndefinedBackendException())
public b_generate_captured_shape, b_generate_captured_shapes
b_generate_captured_shape(b::Backend, s::Shape) = throw(UndefinedBackendException())
b_generate_captured_shapes(b::Backend, ss::Shapes) = throw(UndefinedBackendException())

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
  # No shapes to watch -> never waiting; lets on_change's loop exit cleanly instead of indexing shapes[1] (cf. bounding_box's isempty guard).
  isempty(shapes) ? false :
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
            # Let a real delete failure propagate (delete_shapes' own @defcbs wrappers already retire a
            # dead transport backend and rethrow genuine bugs) rather than swallowing it and rebuilding
            # over leaked refs.
            delete_shapes(shapes)
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
  for s in traced_shapes()
    if ref_value(b, s) == r # Found it!
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
export stroke
public b_stroke
stroke(path;
    material::Material=default_curve_material(),
	  backend::Backend=top_backend(),
	  backends::Backends=(backend,)) =
  for backend in backends
    let mat = material_ref(backend, material)
      b_stroke(backend, path, mat)
    end
  end
export fill
public b_fill
fill(path::GeometryElement;
    material::Material=default_surface_material(),
    backend::Backend=top_backend(),
    backends::Backends=(backend,)) =
  for backend in backends
    let mat = material_ref(backend, material)
      b_fill(backend, path, mat)
    end
  end


###############################################################
# GUI
@defcb(gui_create(name))
@defcb(gui_visible(gui, isvisible))
@defcb(gui_add_folder(gui, name, closed=false))
@defcb(gui_add_button(gui, name, handler))
@defcb(gui_add_checkbox(gui, name, curr, handler))
@defcb(gui_add_slider(gui, name, min, max, step, curr, handler))
@defcb(gui_add_dropdown(gui, name, options, curr, handler))
@defcb(gui_add_slider_parameter(gui, name, min, max, step, parameter))
@defcb(gui_add_button_load_file(gui, name, handler))
@defcb(gui_remove(gui))
