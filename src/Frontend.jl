param_data(p) =
  p isa Symbol ?
    (p, :Any, missing) :
    p isa Expr ?
      (p.head == :kw ?
         (param_data(p.args[1])[1:2]..., p.args[2]) :
         p.head == :(::) ?
           (p.args..., missing) :
           error("Unknown syntax $p")) :
      error("Unknown syntax $p")

def_data(expr) =
  let (name_params, body) =
        expr isa Expr ?
          (expr.head == :(=) ?
             expr.args :
             (expr.head == :call ?
                (expr, :(throw(UndefinedBackendException()))) :
                error("Unknown syntax $p"))) :
          error("Unknown syntax $p")
    name_params.args[1], name_params.args[2:end], body
  end

# Define for (just the) current backend
macro defcb(expr)
  name, params, body = def_data(expr)
  params_data = map(param_data, params)
  pnames = map(pd->pd[1], params_data)
  backend_name = Symbol("b_", name)
  esc(
    quote
      export $(name), $(backend_name)
      # We don't include types in the parameters to avoid multiple dispatch ambiguities.
      # However, it might be interesting to include them in the body as assertions.
      @named_params $(name)($(params...), backend::Backend=top_backend()) =
          $(backend_name)(backend, $(pnames...))
          # We use a default definition for a Any backend to avoid conflict with a similar def in Backend.jl
#      $(backend_name)(backend::Any, $(map(name_typ_init->Expr(:(::), name_typ_init[1], name_typ_init[2]), params_data)...)) =
      $(backend_name)(backend::Any, $(pnames...)) =
          $(body)
      $(name)
    end)
end

# Define for (all the) current backends
macro defcbs(expr)
  name, params, body = def_data(expr)
  params_data = map(param_data, params)
  pnames = map(pd->pd[1], params_data)
  backend_name = Symbol("b_", name)
  esc(
    quote
      export $(name), $(backend_name)
      # We don't include types in the parameters to avoid multiple dispatch ambiguities.
      # However, it might be interesting to include them in the body as assertions.
      @named_params $(name)($(params...), backends::Backends=current_backends()) =
        for backend in backends
          $(backend_name)(backend, $(pnames...))
        end
        # We use a default definition for a Any backend to avoid conflict with a similar def in Backend.jl
#      $(backend_name)(backend::Any, $(map(name_typ_init->Expr(:(::), name_typ_init[1], name_typ_init[2]), params_data)...)) =
      $(backend_name)(backend::Any, $(pnames...)) =
        $(body)
      $(name)
    end)
end

#=
@macroexpand @defcbs foo() =  x + 1
@macroexpand @defcbs foo(bar)
@macroexpand @defcbs foo(bar::Baz)
@macroexpand @defcbs foo(bar=quux)
@macroexpand @defcbs foo(bar::Baz=quux)
=#

# backend define

# backend call
macro bcall(backend, name_args)
  name = name_args.args[1]
  args = name_args.args[2:end]
  backend_name = esc(Symbol("b_$name"))
  quote
    $(backend_name)($(esc(backend)), $(esc.(args)...))
  end
end

# backends call
macro bscall(backends, name_args)
  name = name_args.args[1]
  args = name_args.args[2:end]
  backend_name = esc(Symbol("b_$name"))
  quote
    for backend in $(esc(backends))
      $(backend_name)(backend, $(esc.(args)...))
    end
  end
end

# current backend call
macro cbcall(name_args)
  name = name_args.args[1]
  args = name_args.args[2:end]
  backend_name = esc(Symbol("b_$name"))
  quote
	$(backend_name)(top_backend(), $(esc.(args)...))
  end
end

# current backends call
macro cbscall(name_args)
  name = name_args.args[1]
  args = name_args.args[2:end]
  backend_name = esc(Symbol("b_$name"))
  quote
    for backend in current_backends()
	    $(backend_name)(backend, $(esc.(args)...))
    end
  end
end

@defcbs delete_all_refs()
@defcbs unhighlight_all_refs()
#=
export current_layer
current_layer(backends::Backends=current_backends()) =
  [b_current_layer(b) for b in backends]
current_layer(layer, backends::Backends=current_backends()) =
  for (b, l) in zip(backends, layer)
	b_current_layer(b, l)
  end
=#
@defcbs set_layer_active(layer, status)
@defcbs switch_to_layer(layer)

@defcb disable_update()
@defcb enable_update()

@defcbs set_view(camera::Loc, target::Loc, lens::Real=50, aperture::Real=32)
@defcb get_view()
@defcbs b_zoom_extents()
@defcbs set_view_top()

@defcbs zoom_extents()
#@defcb get_material(ref::Any)
#@defcbs create_material(name::String)
#@defcb current_material()
#@defcbs current_material(material)
@defcbs set_normal_sky()
@defcbs set_overcast_sky()

#=
Renders
There are three major types of renders:
 - Realistic (probably, with realistic colors)
 - White (similar to clay models)
 - Black (similar to clay models with a black background)

 For this to operate properly, rendering must be divided into a series of steps:
 1. Decide the kind of render
 2. Initial setup previous to the render (e.g. choose default materials)
 3. Generate the geometry
 4. Final setup previous to the render (e.g., create sky or a ground plane)
 5. Render
 6. Save the generated image (e.g., in PNG, JPG, PDF)
=#
export render_kind, render_setup, render_view

const render_kind = Parameter(:realistic) # or :white or :black

render_kind_dir_from_render_kind(kind) =
  kind == :realistic ? "Render" :
  kind == :black ? "RenderBlack" : 
  kind == :white ? "RenderWhite" :
  error("Unknown kind $kind")

render_setup(kind::Symbol=:realistic, backend::Backend=top_backend()) = begin
  render_kind_dir(render_kind_dir_from_render_kind(kind))
  render_kind(kind)
  b_render_initial_setup(backend, kind)
end

render_view(name::String="View", backend::Backend=top_backend()) =
  b_render_view(backend, name)
