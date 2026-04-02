#=
Khepri backends need to implement a layered set of virtual operations that
support the higher-level user-accessible operations. Each backend needs to
directly implement a subset of these operations in order to be usable by Khepri.

These operations are layered because different backends might want to support
them at different levels of abstraction. For example, a CAD backend might want
to support the creation of cylinders but not columns, while a BIM backend might
want to directly support the creation of columns. To handle both these cases,
Khepri provides a higher-level default column operation that relies on a
lower-level cylinder operation.

This is where these operations are specified.

Backends are types parameterized by a key identifying the backend (e.g.,
AutoCAD) and by the type of reference they use
=#

abstract type Backend{K,T} end
backend_name(b::Backend) = string(typeof(b))
show(io::IO, b::Backend) = print(io, backend_name(b))


# Backends need to implement operations or an exception is triggered
struct UnimplementedBackendOperationException <: Exception
  backend
  operation
  args
end
showerror(io::IO, e::UnimplementedBackendOperationException) =
  print(io,
    "Operation $(e.operation) is not implemented in backend $(backend_name(e.backend)).\n",
    "Implement $(e.operation)(b::$(typeof(e.backend)), ...) to add support.\n",
    "Alternatively, use a backend that supports this operation.")

missing_specialization(b::Backend, oper=:unknown_operation, args...) =
  error(UnimplementedBackendOperationException(b, oper, args))

macro bdef(call)
  name, escname, params = call.args[1], esc(call.args[1]), esc.(call.args[2:end])
  quote
    export $(escname)
    $(escname)(b::Backend, $(params...)) =
      missing_specialization(b, $(QuoteNode(name)), $(params...))
  end
end

@bdef(void_ref())

export new_refs
new_refs(b::Backend{K,T}) where {K,T} = T[]

#=
@defbackend generates the standard type infrastructure for a Khepri backend:
- Key type, Id alias, Ref/NativeRef aliases
- Backend struct with refs and transaction fields (plus user extra fields)
- Type alias (e.g., const TBS = ThebesBackend)
- void_ref, backend_name, and optional view_type implementations
- Property forwarding for mixin fields (getproperty/setproperty!/propertynames)
- Default b_* operation implementations for known mixins

Usage:
  @defbackend Name Alias begin
    id_type = Int              # optional, default: Int
    void_ref = 0               # optional, default: 0
    view_type = FrontendView() # optional, default: nothing (use BackendView)
    parent = LazyBackend       # optional, default: Backend
    mixin(local_shapes)        # shapes, current_layer, layers
    mixin(render_state)        # date, place, render_env, ground_level, ground_material
    mixin(io)                  # io::IOBuffer
    # Extra struct fields:
    drawing::Any = nothing
    next_id::Int = 1
  end
=#

# Helper: generate getproperty/setproperty!/propertynames for mixin forwarding.
# forwarded is a Vector of (field_symbol, mixin_field_symbol) pairs.
function _defbackend_property_forwarding(ealias, forwarded, direct_names)
  isempty(forwarded) && return nothing
  # Build getproperty chain
  get_branches = map(forwarded) do (field_sym, mixin_field)
    :(sym === $(QuoteNode(field_sym)) && return getfield(getfield(b, $(QuoteNode(mixin_field))), $(QuoteNode(field_sym))))
  end
  set_branches = map(forwarded) do (field_sym, mixin_field)
    # Use convert to match Julia's default setproperty! behavior
    :(sym === $(QuoteNode(field_sym)) && return let obj = getfield(b, $(QuoteNode(mixin_field)))
        setfield!(obj, $(QuoteNode(field_sym)), convert(fieldtype(typeof(obj), $(QuoteNode(field_sym))), val))
      end)
  end
  forwarded_names = Tuple(first.(forwarded))
  all_names = (forwarded_names..., direct_names...)
  quote
    function Base.getproperty(b::$(ealias), sym::Symbol)
      $(get_branches...)
      getfield(b, sym)
    end
    function Base.setproperty!(b::$(ealias), sym::Symbol, val)
      $(set_branches...)
      setfield!(b, sym, val)
    end
    Base.propertynames(b::$(ealias)) = $(all_names)
  end
end

macro defbackend(name, alias, body)
  name_str = string(name)
  key_sym = Symbol(name_str, "Key")
  id_sym = Symbol(name_str, "Id")
  ref_sym = Symbol(name_str, "Ref")
  nref_sym = Symbol(name_str, "NativeRef")
  backend_sym = Symbol(name_str, "Backend")
  # Parse the begin...end block
  @assert body.head == :block "Expected begin...end block"
  id_type = :Int
  void_ref_val = 0
  view_type_val = nothing
  parent_type = nothing
  extra_fields = []
  mixins = Symbol[]
  for expr in body.args
    expr isa LineNumberNode && continue
    if expr isa Expr && expr.head == :(=)
      lhs = expr.args[1]
      rhs = expr.args[2]
      if lhs == :id_type
        id_type = rhs
      elseif lhs == :void_ref
        void_ref_val = rhs
      elseif lhs == :view_type
        view_type_val = rhs
      elseif lhs == :parent
        parent_type = rhs
      elseif lhs isa Expr && lhs.head == :(::)
        # field::Type = default
        push!(extra_fields, expr)
      else
        error("@defbackend: unknown option '$lhs'")
      end
    elseif expr isa Expr && expr.head == :(::)
      # field::Type (no default) — not valid for @kwdef
      error("@defbackend: field '$(expr.args[1])' must have a default value")
    elseif expr isa Expr && expr.head == :call && expr.args[1] == :mixin
      # mixin(name)
      mixin_name = expr.args[2]
      mixin_name isa Symbol || error("@defbackend: mixin name must be a symbol, got: $mixin_name")
      haskey(KhepriBase.MIXIN_REGISTRY, mixin_name) || error("@defbackend: unknown mixin '$mixin_name'. Known mixins: $(join(keys(KhepriBase.MIXIN_REGISTRY), ", "))")
      push!(mixins, mixin_name)
    else
      error("@defbackend: unexpected expression: $expr")
    end
  end

  # Build mixin struct fields and forwarding info
  mixin_struct_fields = Expr[]
  forwarded = Tuple{Symbol, Symbol}[]  # (field_name, mixin_container_field)
  for mx in mixins
    info = KhepriBase.MIXIN_REGISTRY[mx]
    mixin_field = info.field
    mixin_type = info.type
    # Add the composed struct field: _local_shapes::LocalShapes = LocalShapes()
    push!(mixin_struct_fields, :($(mixin_field)::$(mixin_type) = $(mixin_type)()))
    for field_name in info.fields
      push!(forwarded, (field_name, mixin_field))
    end
  end

  # Determine transaction type: ManualCommit if local_shapes present, AutoCommit otherwise
  has_local_shapes = :local_shapes in mixins
  transaction_default = has_local_shapes ?
    :(Parameter{KhepriBase.Transaction}(KhepriBase.ManualCommitTransaction())) :
    :(Parameter{KhepriBase.Transaction}(KhepriBase.AutoCommitTransaction()))

  # Determine parent type
  ekey = esc(key_sym)
  eid = esc(id_sym)
  eref = esc(ref_sym)
  enref = esc(nref_sym)
  ebackend = esc(backend_sym)
  ealias = esc(alias)
  eid_type = esc(id_type)
  evoid = esc(void_ref_val)

  parent_expr = if isnothing(parent_type)
    :(Backend{$(ekey), $(eid)})
  else
    let ept = esc(parent_type)
      :($(ept){$(ekey), $(eid)})
    end
  end

  extra_struct_fields = map(esc, extra_fields)
  mixin_struct_fields_esc = map(esc, mixin_struct_fields)

  view_expr = isnothing(view_type_val) ? nothing :
    :(KhepriBase.view_type(::Type{$(ealias)}) = $(esc(view_type_val)))

  # Collect direct (non-underscore-prefixed) field names for propertynames
  direct_field_names = Symbol[:refs, :transaction]
  for f in extra_fields
    # f is `field::Type = default`, lhs is `field::Type`
    lhs = f.args[1]
    push!(direct_field_names, lhs.args[1])
  end

  # Generate property forwarding
  forwarding_expr = _defbackend_property_forwarding(ealias, forwarded, Tuple(direct_field_names))

  quote
    abstract type $(ekey) end
    const $(eid) = $(eid_type)
    const $(eref) = GenericRef{$(ekey), $(eid)}
    const $(enref) = NativeRef{$(ekey), $(eid)}
    Base.@kwdef mutable struct $(ebackend) <: $(parent_expr)
      refs::References{$(ekey), $(eid)} = References{$(ekey), $(eid)}()
      transaction::Parameter{KhepriBase.Transaction} = $(transaction_default)
      $(mixin_struct_fields_esc...)
      $(extra_struct_fields...)
    end
    const $(ealias) = $(ebackend)
    export $(ealias), $(ebackend), $(ekey), $(eid), $(eref), $(enref)
    KhepriBase.void_ref(b::$(ealias)) = $(evoid)
    KhepriBase.backend_name(b::$(ealias)) = $(name_str)
    $(view_expr)
    $(forwarding_expr)
  end
end
export @defbackend

# Safely extend refs from a b_* operation that may return a vector or a scalar.
collect_ref!(refs, r::AbstractVector) = append!(refs, r)
collect_ref!(refs, r) = push!(refs, r)

############################################################
# Zeroth tier: curves. Not all backends support these.

export b_point, b_line, b_closed_line, b_polygon, b_regular_polygon,
       b_nurbs_curve,
       b_spline, b_closed_spline, b_circle, b_arc, b_ellipse, b_rectangle

@bdef(b_point(p, mat))

@bdef(b_line(ps, mat))

b_polygon(b::Backend, ps, mat) =
  b_line(b, [ps..., ps[1]], mat)

# Legacy
const b_closed_line = b_polygon

b_regular_polygon(b::Backend, edges, c, r, angle, inscribed, mat) =
  b_polygon(b, regular_polygon_vertices(edges, c, r, angle, inscribed), mat)

b_nurbs_curve(b::Backend, ps, order, cps, knots, weights, closed, mat) =
  # Sample the curve at many points for a visually smooth polyline
  let ci = curve_interpolator(ps, closed),
      n = max(length(ps) * 16, 64),
      sampled = [location_at(ci, t) for t in division(0, 1, n, !closed)]
    closed ?
      b_polygon(b, sampled, mat) :
      b_line(b, sampled, mat)
  end

b_spline(b::Backend, ps, mat) =
  b_spline(b, ps, false, false, mat)

b_spline(b::Backend, ps, v0, v1, mat) =
  if !(v0 isa Vec) && !(v1 isa Vec)
    let ci = curve_interpolator(ps, false),
        cps = curve_control_points(ci),
        n = length(cps),
        knots = curve_knots(ci)
      b_nurbs_curve(b, ps, 5, cps, knots, fill(1.0, n), false, mat)
    end
  else
    # Cubic Hermite spline with endpoint tangent constraints
    let coords = [raw_point(p) for p in ps],
        n = length(coords),
        # Compute tangents: Catmull-Rom for interior, user-specified for endpoints
        tangents = [
          (v0 isa Vec ?
            let v = raw_point(in_world(v0)),
                d = sqrt(sum((coords[2][j] - coords[1][j])^2 for j in 1:3))
              (v[1]*d, v[2]*d, v[3]*d) end :
            ((coords[2][j] - coords[1][j]) for j in 1:3) |> Tuple);
          [((coords[i+1][j] - coords[i-1][j]) / 2 for j in 1:3) |> Tuple
           for i in 2:n-1];
          (v1 isa Vec ?
            let v = raw_point(in_world(v1)),
                d = sqrt(sum((coords[n][j] - coords[n-1][j])^2 for j in 1:3))
              (v[1]*d, v[2]*d, v[3]*d) end :
            ((coords[n][j] - coords[n-1][j]) for j in 1:3) |> Tuple)],
        # Sample segments proportionally to chord length
        seg_lengths = [sqrt(sum((coords[i+1][j] - coords[i][j])^2 for j in 1:3)) for i in 1:n-1],
        total_length = sum(seg_lengths),
        n_total = max(n * 16, 64),
        sampled = let result = [xyz(coords[1]..., world_cs)]
          for seg in 1:n-1
            n_seg = max(round(Int, n_total * seg_lengths[seg] / total_length), 2)
            p0, p1 = coords[seg], coords[seg+1]
            m0, m1 = tangents[seg], tangents[seg+1]
            for k in 1:n_seg
              t = k / n_seg
              h00 = 2t^3 - 3t^2 + 1
              h10 = t^3 - 2t^2 + t
              h01 = -2t^3 + 3t^2
              h11 = t^3 - t^2
              push!(result, xyz(
                h00*p0[1] + h10*m0[1] + h01*p1[1] + h11*m1[1],
                h00*p0[2] + h10*m0[2] + h01*p1[2] + h11*m1[2],
                h00*p0[3] + h10*m0[3] + h01*p1[3] + h11*m1[3],
                world_cs))
            end
          end
          result
        end
      b_line(b, sampled, mat)
    end
  end

b_closed_spline(b::Backend, ps, mat) =
  let ci = curve_interpolator(ps, true),
      cps = curve_control_points(ci),
      n = length(cps),
      knots = curve_knots(ci)
    b_nurbs_curve(b, ps, 5, cps, knots, fill(1.0, n), true, mat)
  end

b_circle(b::Backend, c, r, mat) =
  b_closed_spline(b, regular_polygon_vertices(32, c, r, 0, true), mat)

b_arc(b::Backend, c, r, α, Δα, mat) =
  Δα ≈ 0.0 ?
    void_ref(b) :
    let pts = [c + vpol(r, a, c.cs)
               for a in division(α, α + Δα, max(ceil(Int, Δα*32/2/π), 2), true)]
      b_spline(b, pts, nothing, nothing, mat)
    end

b_ellipse(b::Backend, c, rx, ry, mat) =
  b_closed_spline(b,
    [add_xy(c, rx*cos(ϕ), ry*sin(ϕ))
     for ϕ in division(0, 2pi, 64, false)], mat)
  
b_rectangle(b::Backend, c, dx, dy, mat) =
  b_polygon(b, [c, add_x(c, dx), add_xy(c, dx, dy), add_y(c, dy)], mat)

#############################################################
# First tier: everything is a triangle or a set of triangles
export b_trig, b_quad, b_ngon, b_quad_strip, b_quad_strip_closed, b_strip

@bdef(b_trig(p1, p2, p3))

# By default, we silently drop the mat
b_trig(b::Backend, p1, p2, p3, mat) =
  b_trig(b, p1, p2, p3)

b_quad(b::Backend, p1, p2, p3, p4, mat) =
  [b_trig(b, p1, p2, p3, mat),
   b_trig(b, p1, p3, p4, mat)]

b_ngon(b::Backend, ps, pivot, smooth, mat) =
  [(b_trig(b, pivot, ps[i], ps[i+1], mat)
    for i in 1:size(ps,1)-1)...,
   b_trig(b, pivot, ps[end], ps[1], mat)]

b_quad_strip(b::Backend, ps, qs, smooth, mat) =
  vcat([b_quad(b, ps[i], ps[i+1], qs[i+1], qs[i], mat) for i in 1:size(ps,1)-1]...)

b_quad_strip_closed(b::Backend, ps, qs, smooth, mat) =
  b_quad_strip(b, [ps..., ps[1]], [qs..., qs[1]], smooth, mat)

b_strip(b::Backend, path1, path2, mat) =
  let v1s = path_vertices(path1),
      v2s = path_vertices(path2)
    length(v1s) != length(v2s) ?
      error("Paths with different resolution ($(length(v1s)) vs $(length(v2s)))") :
      is_closed_path(path1) && is_closed_path(path2) ?
        b_quad_strip_closed(
          b, v1s, v2s, is_smooth_path(path1) || is_smooth_path(path2), mat) :
        b_quad_strip(
          b, v1s, v2s, is_smooth_path(path1) || is_smooth_path(path2), mat)
  end

b_strip(b::Backend, path1::Region, path2::Region, mat) =
  [b_strip(b, outer_path(path1), outer_path(path2), mat),
   [b_strip(b, p2, p1, mat) for (p1, p2) in zip(inner_paths(path1), inner_paths(path2))]...
  ]

############################################################
# Second tier: surfaces
export b_surface_polygon, b_surface_polygon_with_holes,
       b_surface_regular_polygon, b_surface_rectangle,
       b_surface_circle, b_surface_ring, b_surface_arc, b_surface_ellipse, b_surface_closed_spline,
       b_surface, b_surface_grid, b_smooth_surface_grid, b_surface_mesh

# Ear clipping triangulation for simple (possibly concave) polygons.
# Takes a vector of 3D coordinates (anything indexable with [1],[2],[3])
# and returns a vector of (i,j,k) index triples.
export triangulate_polygon
function triangulate_polygon(coords)
  n = length(coords)
  n < 3 && return Tuple{Int,Int,Int}[]
  n == 3 && return [(1, 2, 3)]
  # Project to 2D using the polygon normal (Newell's method)
  nx, ny, nz = 0.0, 0.0, 0.0
  for i in 1:n
    j = mod(i, n) + 1
    x1, y1, z1 = Float64(coords[i][1]), Float64(coords[i][2]), Float64(coords[i][3])
    x2, y2, z2 = Float64(coords[j][1]), Float64(coords[j][2]), Float64(coords[j][3])
    nx += (y1 - y2) * (z1 + z2)
    ny += (z1 - z2) * (x1 + x2)
    nz += (x1 - x2) * (y1 + y2)
  end
  ax, ay, az = abs(nx), abs(ny), abs(nz)
  pts = if ax >= ay && ax >= az
    [(Float64(c[2]), Float64(c[3])) for c in coords]
  elseif ay >= ax && ay >= az
    [(Float64(c[1]), Float64(c[3])) for c in coords]
  else
    [(Float64(c[1]), Float64(c[2])) for c in coords]
  end
  # Signed area determines winding
  area = 0.0
  for i in 1:n
    j = mod(i, n) + 1
    area += pts[i][1] * pts[j][2] - pts[j][1] * pts[i][2]
  end
  ccw = area > 0
  # Ear clipping
  idx = collect(1:n)
  trigs = Tuple{Int,Int,Int}[]
  sizehint!(trigs, n - 2)
  while length(idx) > 3
    m = length(idx)
    found = false
    for ii in 1:m
      ia = idx[mod1(ii - 1, m)]
      ib = idx[ii]
      ic = idx[mod1(ii + 1, m)]
      pa, pb, pc = pts[ia], pts[ib], pts[ic]
      cross = (pb[1] - pa[1]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[1] - pa[1])
      (ccw ? cross > 0 : cross < 0) || continue
      inside = false
      for jj in 1:m
        vi = idx[jj]
        vi == ia || vi == ib || vi == ic || begin
          pt = pts[vi]
          d1 = (pt[1] - pb[1]) * (pa[2] - pb[2]) - (pa[1] - pb[1]) * (pt[2] - pb[2])
          d2 = (pt[1] - pc[1]) * (pb[2] - pc[2]) - (pb[1] - pc[1]) * (pt[2] - pc[2])
          d3 = (pt[1] - pa[1]) * (pc[2] - pa[2]) - (pc[1] - pa[1]) * (pt[2] - pa[2])
          if !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0))
            inside = true
          end
          false
        end
        inside && break
      end
      if !inside
        push!(trigs, (ia, ib, ic))
        deleteat!(idx, ii)
        found = true
        break
      end
    end
    if !found
      # Degenerate polygon: fall back to fan triangulation
      for i in 2:length(idx)-1
        push!(trigs, (idx[1], idx[i], idx[i+1]))
      end
      break
    end
  end
  length(idx) == 3 && push!(trigs, (idx[1], idx[2], idx[3]))
  trigs
end

@bdef(b_surface(frontier, mat))

b_surface_polygon(b::Backend, ps, mat) =
  let trigs = triangulate_polygon([raw_point(p) for p in ps])
    [b_trig(b, ps[i], ps[j], ps[k], mat) for (i, j, k) in trigs]
  end

b_surface_polygon_with_holes(b::Backend, ps, qss, mat) =
  # By default, we use half-edges
  b_surface_polygon(b, foldl(subtract_polygon_vertices, qss, init=ps), mat)

b_surface_polygon_with_holes(b::Backend, ps, qss, smooths, mat) =
  b_surface_polygon_with_holes(b, ps, qss, mat)

b_surface_rectangle(b::Backend, c, dx, dy, mat) =
  b_quad(b, c, add_x(c, dx), add_xy(c, dx, dy), add_y(c, dy), mat)

b_surface_regular_polygon(b::Backend, edges, c, r, angle, inscribed, mat) =
  b_ngon(b, regular_polygon_vertices(edges, c, r, angle, inscribed), c, false, mat)

b_surface_circle(b::Backend, c, r, mat) =
  b_surface_regular_polygon(b, 32, c, r, 0, true, mat)

b_surface_ring(b::Backend, c, ri, ro, mat) =
  b_surface_polygon_with_holes(
    b,
    regular_polygon_vertices(64, c, ro, 0, true),
    [regular_polygon_vertices(64, c, ri, 0, true)],
    mat)

b_surface_arc(b::Backend, c, r, α, Δα, mat) =
  b_ngon(b,
         [c + vpol(r, a, c.cs)
          for a in division(α, α + Δα, Δα*32/2/π, true)],
         c, false, mat)

b_surface_ellipse(b::Backend, c, rx, ry, mat) =
  b_surface_closed_spline(b,
    [add_xy(c, rx*cos(ϕ), ry*sin(ϕ))
     for ϕ in division(0, 2pi, 64, false)], mat)

@bdef(b_surface_closed_spline(ps, mat))

#=
This implementation is based on independent quadstrips, therefore, it is impossible
to be smooth in both u,v dimensions. We opt to be smooth in just the v dimension.
=#

b_surface_grid(b::Backend, ptss, closed_u, closed_v, smooth_u, smooth_v, mat) =
  let ptss = maybe_interpolate_grid(ptss, smooth_u, smooth_v),
      (nu, nv) = size(ptss)
    closed_v ?
      vcat([b_quad_strip_closed(b, ptss[i,:], ptss[i+1,:], smooth_v, mat) for i in 1:nu-1]...,
           (closed_u ? [b_quad_strip_closed(b, ptss[end,:], ptss[1,:], smooth_v, mat)] : new_refs(b))...) :
      vcat([b_quad_strip(b, ptss[i,:], ptss[i+1,:], smooth_v, mat) for i in 1:nu-1]...,
           (closed_u ? [b_quad_strip(b, ptss[end,:], ptss[1,:], smooth_v, mat)] : new_refs(b))...)
  end

export maybe_interpolate_grid
maybe_interpolate_grid(ptss, smooth_u, smooth_v) =
  smooth_u || smooth_v ? 
    let interpolator = grid_interpolator(ptss),
        (nu, nv) = size(ptss)
      [location_at(interpolator, u, v)
       for u in division(0, 1, smooth_u ? 4*nu-7 : nu-1),
           v in division(0, 1, smooth_v ? 4*nv-7 : nv-1)] # 2->1, 3->5, 4->9, 5->13
    end :
    ptss

b_smooth_surface_grid(b::Backend, ptss, closed_u, closed_v, mat) =
  b_surface_grid(b, smooth_grid(ptss), closed_u, closed_v, true, true, mat)

b_surface_mesh(b::Backend, vertices, faces, mat) =
  map(faces) do face
    if length(face) == 3
      b_trig(b, vertices[face]..., mat)
      elseif length(face) == 4
        b_quad(b, vertices[face]..., mat)
      else
      b_surface_polygon(b, vertices[face], mat)
      end
    end

# Parametric surface
#=
parametric {
    function { sin(u)*cos(v) }
    function { sin(u)*sin(v) }
    function { cos(u) }

    <0,0>, <2*pi,pi>
    contained_by { sphere{0, 1.1} }
    max_gradient ??
    accuracy 0.0001
    precompute 10 x,y,z
    pigment {rgb 1}
  }
=#

############################################################
# Third tier: solids

#=
We will use trigs, quads, quad_strips, and so on, but in order
to make a solid in the end, we will rely on a solidify operation that, 
by default, does nothing. However, on backends where solids do exist,
the solidify operation should convert those primitive shapes into a 
proper solid.
=#

export b_generic_pyramid_frustum, b_generic_pyramid, b_generic_prism,
       b_generic_pyramid_frustum_with_holes, b_generic_prism_with_holes,
       b_pyramid_frustum, b_pyramid, b_prism,
       b_regular_pyramid_frustum, b_regular_pyramid, b_regular_prism,
       b_cylinder,
       b_cuboid,
       b_box,
       b_sphere,
       b_cone_frustum, b_cone,
       b_torus,
       b_solidify

b_solidify(b::Backend, refs) = refs

# Each solid can have just one material or multiple materials
b_generic_pyramid_frustum(b::Backend, bs, ts, smooth, bmat, tmat, smat) =
  b_solidify(b,
    vcat(isnothing(bmat) ? new_refs(b) : b_surface_polygon(b, reverse(bs), bmat),
         b_quad_strip_closed(b, bs, ts, smooth, smat),
         isnothing(tmat) ? new_refs(b) : b_surface_polygon(b, ts, tmat)))

b_generic_pyramid_frustum_with_holes(b::Backend, bs, ts, smooth, bbs, tts, smooths, bmat, tmat, smat) =
  b_solidify(b,
    [b_surface_polygon_with_holes(b, reverse(bs), bbs, bmat),
     b_quad_strip_closed(b, bs, ts, smooth, smat),
     [b_quad_strip_closed(b, bs, ts, smooth, smat)
      for (bs, ts, smooth) in zip(bbs, tts, smooths)]...,
     b_surface_polygon_with_holes(b, ts, reverse.(tts), tmat)])

b_generic_pyramid(b::Backend, bs, t, smooth, bmat, smat) =
  b_solidify(b,
    vcat(b_surface_polygon(b, reverse(bs), bmat),
         b_ngon(b, bs, t, smooth, smat)))

b_generic_prism(b::Backend, bs, smooth, v, bmat, tmat, smat) =
  b_generic_pyramid_frustum(b, bs, translate(bs, v), smooth, bmat, tmat, smat)
b_generic_prism_with_holes(b::Backend, bs, smooth, bss, smooths, v, bmat, tmat, smat) =
  b_generic_pyramid_frustum_with_holes(b, bs, translate(bs, v), smooth, bss, translate.(bss, v), smooths, bmat, tmat, smat)

b_pyramid_frustum(b::Backend, bs, ts, mat) =
  b_pyramid_frustum(b, bs, ts, mat, mat, mat)

b_pyramid_frustum(b::Backend, bs, ts, bmat, tmat, smat) =
  b_generic_pyramid_frustum(b, bs, ts, false, bmat, tmat, smat)

b_pyramid(b::Backend, bs, t, mat) =
  b_pyramid(b, bs, t, mat, mat)
b_pyramid(b::Backend, bs, t, bmat, smat) =
  b_generic_pyramid(b, bs, t, false, bmat, smat)

b_prism(b::Backend, bs, v, mat) =
  b_prism(b, bs, v, mat, mat, mat)
b_prism(b::Backend, bs, v, bmat, tmat, smat) =
  b_pyramid_frustum(b, bs, translate(bs, v), bmat, tmat, smat)

b_regular_pyramid_frustum(b::Backend, edges, cb, rb, angle, h, rt, inscribed, mat) =
  b_regular_pyramid_frustum(b, edges, cb, rb, angle, h, rt, inscribed, mat, mat, mat)
b_regular_pyramid_frustum(b::Backend, edges, cb, rb, angle, h, rt, inscribed, bmat, tmat, smat) =
  b_pyramid_frustum(
    b,
    regular_polygon_vertices(edges, cb, rb, angle, inscribed),
    regular_polygon_vertices(edges, add_z(cb, h), rt, angle, inscribed),
    bmat, tmat, smat)

b_regular_pyramid(b::Backend, edges, cb, rb, angle, h, inscribed, mat) =
  b_regular_pyramid(b, edges, cb, rb, angle, h, inscribed, mat, mat)
b_regular_pyramid(b::Backend, edges, cb, rb, angle, h, inscribed, bmat, smat) =
  b_pyramid(
    b,
    regular_polygon_vertices(edges, cb, rb, angle, inscribed),
    add_z(cb, h),
    bmat, smat)

b_regular_prism(b::Backend, edges, cb, rb, angle, h, inscribed, mat) =
  b_regular_prism(b, edges, cb, rb, angle, h, inscribed, mat, mat, mat)
b_regular_prism(b::Backend, edges, cb, rb, angle, h, inscribed, bmat, tmat, smat) =
  b_regular_pyramid_frustum(b, edges, cb, rb, angle, h, rb, inscribed, bmat, tmat, smat)

# Tessellation resolution for smooth solids (spheres, cylinders, cones, tori).
# Backends that produce vector output (TikZ, SVG) should override with a lower value.
export tessellation_divisions
tessellation_divisions(b::Backend) = 32

b_cylinder(b::Backend, cb, r, h, mat) =
  b_cylinder(b, cb, r, h, mat, mat, mat)
b_cylinder(b::Backend, cb, r, h, bmat, tmat, smat) =
  let n = tessellation_divisions(b)
    b_generic_prism(
      b,
      regular_polygon_vertices(n, cb, r, 0, true),
      true,
      vz(h, cb.cs),
      bmat, tmat, smat)
  end

b_cuboid(b::Backend, pb0, pb1, pb2, pb3, pt0, pt1, pt2, pt3, mat) =
  b_solidify(b,
    vcat(b_surface_polygon(b, [pb3, pb2, pb1, pb0], mat), # quads do not work with concave quads
         b_quad_strip_closed(b, [pb0, pb1, pb2, pb3], [pt0, pt1, pt2, pt3], false, mat),
         b_surface_polygon(b, [pt0, pt1, pt2, pt3], mat)))

b_box(b::Backend, c, dx, dy, dz, mat) =
  let pb0 = c,
      pb1 = add_x(c, dx),
      pb2 = add_xy(c, dx, dy),
      pb3 = add_y(c, dy),
      pt0 = add_z(pb0, dz),
      pt1 = add_z(pb1, dz),
      pt2 = add_z(pb2, dz),
      pt3 = add_z(pb3, dz)
    b_cuboid(b, pb0, pb1, pb2, pb3, pt0, pt1, pt2, pt3, mat)
  end

b_sphere(b::Backend, c, r, mat) =
  let n = tessellation_divisions(b),
      step = π/n,
      ϕs = division(0, 2π, n, false)
    b_solidify(b,
      [b_ngon(b, [add_sph(c, r, ϕ, step) for ϕ in ϕs], add_sph(c, r, 0, 0), true, mat),
       [b_quad_strip_closed(b,
          [add_sph(c, r, ϕ, ψ+step) for ϕ in ϕs],
          [add_sph(c, r, ϕ, ψ) for ϕ in ϕs],
          true, mat) for ψ in step:step:π-step]...,
       b_ngon(b, reverse!([add_sph(c, r, ϕ, π-step) for ϕ in ϕs]), add_sph(c, r, 0, π), true, mat)])
  end

b_cone(b::Backend, cb, r, h, mat) =
  b_cone(b, cb, r, h, mat, mat)

b_cone(b::Backend, cb, r, h, bmat, smat) =
  b_generic_pyramid(
  b,
  regular_polygon_vertices(tessellation_divisions(b), cb, r, 0, true),
  add_z(cb, h),
  true,
  bmat, smat)

b_cone_frustum(b::Backend, cb, rb, h, rt, mat) =
  b_cone_frustum(b, cb, rb, h, rt, mat, mat, mat)

b_cone_frustum(b::Backend, cb, rb, h, rt, bmat, tmat, smat) =
  let n = tessellation_divisions(b)
    b_generic_pyramid_frustum(
      b,
      regular_polygon_vertices(n, cb, rb, 0, true),
      regular_polygon_vertices(n, add_z(cb, h), rt, 0, true),
      true,
      bmat, tmat, smat)
  end

b_torus(b::Backend, c, ra, rb, mat) =
  let n = tessellation_divisions(b)
    b_surface_grid(
      b,
      [add_sph(add_pol(c, ra, ϕ), rb, ϕ, ψ)
       for ψ in division(0, 2π, n, false), ϕ in division(0, 2π, 2n, false)],
        true, true,
        true, true,
        mat)
  end

export b_mesh_obj_fmt
@bdef(b_mesh_obj_fmt(obj_name, transform))
export b_set_environment
@bdef(b_set_environment(env_name, set_background))

##################################################################
# Paths and Regions
b_surface(b::Backend, path::ClosedPath, mat) =
  b_surface_polygon(b, path_vertices(path), mat)

b_surface(b::Backend, region::Region, mat) =
  b_surface_polygon_with_holes(
    b,
    path_vertices(outer_path(region)),
    [reverse(path_vertices(path)) for path in inner_paths(region)],
    BitVector([is_smooth_path(path) for path in region.paths]),
    mat)

b_surface(b::Backend, frontier::Shapes, mat) =
  let path = foldr(join_paths, [convert(OpenPolygonalPath, shape_path(e)) for e in frontier])
    and_delete_shapes(b_surface_polygon(b, path_vertices(path), mat), frontier)
  end

# In theory, this should be implemented using a loft
b_path_frustum(b::Backend, bpath, tpath, bmat, tmat, smat) =
  let blength = path_length(bpath),
    tlength = path_length(tpath),
    n = max(length(path_vertices(bpath)), length(path_vertices(bpath))),
    bs = division(bpath, n),
    ts = division(tpath, n)
    # We should rotate one of the vertices array to minimize the distance
    # between corresponding so that they align better.
  b_generic_pyramid_frustum(
    b, bs, ts,
    is_smooth_path(bpath) || is_smooth_path(tpath),
    bmat, tmat, smat)
  end

# Extrusions, lofts, sweeps, etc
export b_extruded_point, b_extruded_curve, b_extruded_surface, b_sweep, b_loft

# Extruding a profile
b_extruded_point(b::Backend, path, v, cb, mat) =
  let p = path_on(path, cb).location
    b_line(b, [p, p + v], mat)
  end

@bdef(b_extruded_curve(path, v, cb, mat))

b_extruded_curve(b::Backend, path::OpenPolygonalPath, v, cb, mat) =
  let bs = path_vertices_on(path, cb),
      ts = translate(bs, v)
    b_quad_strip(b, bs, ts, is_smooth_path(path), mat)
  end

b_extruded_curve(b::Backend, path::ClosedPolygonalPath, v, cb, mat) =
  b_extruded_curve(b, convert(OpenPolygonalPath, path), v, cb, mat)

b_extruded_curve(b::Backend, profile::CircularPath, v, cb, mat) =
  v.cs === cb.cs && iszero(v.x) && iszero(v.y) ?
    # HACK: This is wrong. It should be an open cylinder
    b_cylinder(b, add_xy(cb, profile.center.x, profile.center.y), profile.radius, v.z, nothing, nothing, mat) :
  b_generic_prism(b,
    path_vertices_on(profile, cb),
    is_smooth_path(profile),
    v,
    nothing, nothing, mat)

b_extruded_surface(b::Backend, profile, v, cb, mat) =
  b_extruded_surface(b, profile, v, cb, mat, mat, mat)

b_extruded_surface(b::Backend, profile::Region, v, cb, bmat, tmat, smat) =
  let outer = outer_path(profile),
      inners = inner_paths(profile)
    vcat(b_extruded_curve(b, outer, v, cb, smat),
         [b_extruded_curve(b, inner, v, cb, smat) for inner in inners]...,
         b_surface(b, path_on(profile, cb), bmat),
         b_surface(b, translate(path_on(profile, cb), v), tmat))
  end
b_extruded_curve(b::Backend, profile::PathSequence, v, cb, mat) =
  vcat([b_extruded_curve(b, subprofile, v, cb, mat) for subprofile in profile.paths]...)
b_extruded_curve(b::Backend, profile::Path, v, cb, mat) =
  b_extruded_curve(b, convert(OpenPolygonalPath, profile), v, cb, mat)

b_loft(b::Backend, profiles, closed, smooth, mat) =
  let ptss = path_vertices.(profiles),
    n = mapreduce(length, max, ptss),
    vss = map(profile->map_division(identity, profile, n), profiles)
  b_surface_grid(b, hcat(vss...), is_closed_path(profiles[1]), closed, is_smooth_path(profiles[1]), smooth, mat)
  end


b_swept_curve(b::Backend, path, profile, rotation, scaling, mat) =
  b_sweep(b, path, profile, rotation, scaling, mat)

b_swept_surface(b::Backend, path, profile, rotation, scaling, mat) =
  b_sweep(b, path, profile, rotation, scaling, mat)

b_sweep(b::Backend, path, profile::Region, rotation, scaling, mat) =
  let outer = outer_path(profile),
      inners = inner_paths(profile),
      frames = path_interpolated_frames(path),
      frames = rotation == 0 ?
        frames : 
        [loc_from_o_phi(frame, phi) for (frame, phi) in zip(frames, map_division(identity, 0, rotation, length(frames)-1))],
      profiles = map_division(s->scale(profile, s, u0()), 1, scaling, length(frames)-1)
    vcat(b_sweep(b, path, outer, rotation, scaling, mat),
         [b_sweep(b, path, inner, rotation, scaling, mat) for inner in inners]...,
         # HACK
         # If the final profile coincides with the first, then it doesn't make sense
         # to create the closing surfaces
         b_surface(b, path_on(profiles[1], frames[1]), mat),
         b_surface(b, path_on(profiles[end], frames[end]), mat))
  end

b_sweep(b::Backend, path, profile, rotation, scaling, mat) =
  let frames = rotation == 0 ?
                rotation_minimizing_frames(path_frames(path)) :
                let subframes = path_interpolated_frames(path, path_domain(path), 
                                                         collinearity_tolerance(),
                                                         ceil(Int, log(abs(rotation))*4))
                  map(loc_from_o_phi, subframes, map_division(identity, 0, rotation, length(subframes)-1))
                end,
      profiles = map_division(s->scale(profile, s, u0()), 1, scaling, length(frames)-1),
      verticess = map(path_vertices, profiles),
      verticess = is_smooth_path(profile) ?
        let n = mapreduce(length, max, verticess)-1
          map(path->map_division(identity, path, n), profiles)
        end :
        verticess,
      points = hcat(map(on_cs, verticess, frames)...)
    #[b_sphere(b, p, 0.01, mat) for p in points]
    b_surface_grid(
      b,
      points,
      is_closed_path(profile),
      is_closed_path(path),
      is_smooth_path(profile),
      is_smooth_path(path),
      mat)
  end

b_revolved_point(b::Backend, profile, p, n, start_angle, amplitude, mat) =
  let q = profile.position,
      pp = perpendicular_point(p, n, q)
    b_arc(b, loc_from_o_vz(pp, n), distance(pp, q), start_angle, amplitude, mat)
  end

b_revolved_curve(b::Backend, profile, p, n, start_angle, amplitude, mat) = 
  let profile = translate(convert(Path, profile), u0()-p),
      pp = loc_from_o_vz(p, n),
      vertices = path_frames(profile),
      frames = map_division(ϕ -> loc_from_o_phi(pp, start_angle + ϕ), 0, amplitude, ceil(Int, amplitude*10), amplitude < 2pi),
      points = hcat(map(frame->on_cs(vertices, frame), frames)...)
    b_surface_grid(
        b,
        points,
        is_closed_path(profile),
        amplitude >= 2pi,
        is_smooth_path(profile),
        true,
        mat)
  end

b_revolved_surface(b::Backend, profile, p, n, start_angle, amplitude, mat) =
  let profile = convert(Region, profile),
      outer = outer_path(profile),
      inners = inner_paths(profile)
    vcat(b_revolved_curve(b, outer, p, n, start_angle, amplitude, mat),
         [b_revolved_curve(b, inner, p, n, start_angle, amplitude, mat) for inner in inners]...,
         (amplitude < 2pi ?
          let pp = loc_from_o_vz(p, n),
              profile = translate(profile, u0()-p),
              frames = map_division(ϕ -> loc_from_o_phi(pp, start_angle + ϕ), 0, amplitude, 1)
           [b_surface(b, path_on(profile, frames[1]), mat),
            b_surface(b, path_on(profile, frames[end]), mat)]
          end :
          new_refs(b)))
  end


# Booleans

export b_subtracted, b_intersected, b_united,
       b_subtracted_surfaces, b_intersected_surfaces, b_united_surfaces, 
       b_subtracted_solids, b_intersected_solids, b_united_solids,
       b_slice, b_unite_refs, b_slice_ref

@bdef b_subtract_ref(sref, mref)
@bdef b_intersect_ref(sref, mref)
b_unite_ref(b::Backend, sref, mref) = vcat(sref, mref)

b_unite_refs(b::Backend, rs) =
  foldl((s, r)->b_unite_ref(b, s, r), rs)

b_subtracted_surfaces(b::Backend, source, mask, mat) =
  b_subtracted(b, source, mask, mat)

b_subtracted_solids(b::Backend, source, mask, mat) =
  b_subtracted(b, source, mask, mat)

b_subtracted(b::Backend, source, mask, mat) =
  and_mark_deleted(b,
    map_ref(b, source) do s
      and_mark_deleted(b,
        map_ref(b, mask) do r
          b_subtract_ref(b, s, r)
        end,
        [mask])
      end,
    [source])

b_intersected_surfaces(b::Backend, source, mask, mat) =
  b_intersected(b, source, mask, mat)
  
b_intersected_solids(b::Backend, source, mask, mat) =
  b_intersected(b, source, mask, mat)
  
b_intersected(b::Backend, source, mask, mat) =
  and_mark_deleted(b,
    map_ref(b, source) do s
      and_mark_deleted(b,
        map_ref(b, mask) do r
          b_intersect_ref(b, s, r)
        end,
        [mask])
      end,
    [source])

b_united_surfaces(b::Backend, source, mask, mat) =
  b_united(b, source, mask, mat)
    
b_united_solids(b::Backend, source, mask, mat) =
  b_united(b, source, mask, mat)
  
b_united(b::Backend, source, mask, mat) =
  and_mark_deleted(b,
    b_unite_refs(b, vcat(ref_values(b, source), ref_values(b, mask))),
    [source, mask])

b_slice(b::Backend, shape, p, v, mat) =
  and_mark_deleted(b,
    map_ref(b, shape) do s
      b_slice_ref(b, s, p, v)
    end,
    [shape])

b_slice_ref(b::Backend, r, p, v) =
  b_subtract_ref(b, r, b_regular_prism(b, 4, loc_from_o_vz(p, v), 1e5, 0, 1e5, true, void_ref(b)))

# Stroke and fill operations over paths
# This is specially useful for debuging

b_stroke(b::Backend, path::CircularPath, mat) =
  b_circle(b, path.center, path.radius, mat)
b_stroke(b::Backend, path::RectangularPath, mat) =
  b_rectangle(b, path.corner, path.dx, path.dy, mat)
b_stroke(b::Backend, path::ArcPath, mat) =
  b_arc(b, path.center, path.radius, path.start_angle, path.amplitude, mat)
b_stroke(b::Backend, path::OpenPolygonalPath, mat) =
  b_line(b, path.vertices, mat)
b_stroke(b::Backend, path::ClosedPolygonalPath, mat) =
  b_polygon(b, path.vertices, mat)
b_stroke(b::Backend, path::OpenSplinePath, mat) =
  b_spline(b, path.vertices, path.v0, path.v1, mat)
b_stroke(b::Backend, path::ClosedSplinePath, mat) =
  b_closed_spline(b, path.vertices, mat)
b_stroke(b::Backend, path::Region, mat) =
  [b_stroke(b, path, mat) for path in path.paths]
b_stroke(b::Backend, path::Mesh, mat) =
  let vs = path.vertices
    for face in path.faces
      b_line(b, vs[face.+1], mat) #1-indexed
    end
  end
b_stroke(b::Backend, path::PathSequence, mat) =
  for path in path.paths
    b_stroke(b, path, mat)
  end

b_fill(b::Backend, path::CircularPath, mat) =
  b_surface_circle(b, path.center, path.radius, mat)
b_fill(b::Backend, path::RectangularPath, mat) =
  b_surface_rectangle(b, path.corner, path.dx, path.dy, mat)
b_fill(b::Backend, path::ClosedPolygonalPath, mat) =
  b_surface_polygon(b, path.vertices, mat)
b_fill(b::Backend, path::ClosedSplinePath, mat) =
  b_surface_closed_spline(b, path.vertices, mat)
b_fill(b::Backend, path::Region, mat) =
  b_surface(b, path, mat)
b_fill(b::Backend, path::Mesh, mat) =
  b_surface_mesh(b, path.vertices, path.faces, mat)

export b_realize_path
b_realize_path(b::Backend, path::Region, mat) =
  b_fill(b, path, mat)
b_realize_path(b::Backend, path, mat) =
  b_stroke(b, path, mat)

##################################################################
# Dimensions
#=
Technical drawings need dimensions.
We need extension lines and dimension lines (with or without arrows)
and with a text.
Guidelines say that an extension line should be start 1mm away from
the object and extend 2mm beyond the dimension line. This all depends
on the scale we are using so I'll include a size parameter and I'll
consider the inicial spacing as 5% of the size and the extension to
be extra 10% in excess of the size.
=#
export b_dimension, b_ext_line, b_dim_line, b_text, b_text_size, b_arc_dimension

b_dimension(b::Backend, p, q, str, size, offset, mat) =
  let qp = in_world(q - p),
      phi = pol_phi(qp),
      outside = pi/2 <= phi <= 3pi/2,
      v = vpol(outside ? size : 2*size, phi-pi/2),
      uv = unitized(v),
      (si, se) = (offset*size, 2*offset*size),
      (vi, ve) = (uv*si, uv*se),
      (tp, tq, tv) = outside ? (q, p, vpol(1, phi + pi)) : (p, q, vpol(1, phi))
    offset == 0 ?
      b_dim_line(b, tp, tq, tv, str, size, outside, mat) :
      [b_ext_line(b, p + vi, p + v + ve, mat),
       b_ext_line(b, q + vi, q + v + ve, mat),
       b_dim_line(b, tp + v, tq + v, tv, str, size, outside, mat)]
  end
b_ext_line(b::Backend, p, q, mat) =
  b_line(b, [p, q], mat)
b_dim_line(b::Backend, p, q, tv, str, size, outside, mat) =
  let (minx, maxx, miny, maxy) = b_text_size(b, str, size, mat),
    tp = p + tv*((distance(p, q)-(maxx-minx))/2)
    [b_line(b, [p, q], mat),
     b_text(b, str, add_y(loc_from_o_vx(tp, tv), size*0.1-miny), size, mat)]
  end

b_arc_dimension(b::Backend, c, r, α, Δα, rstr, dstr, size, offset, mat) =
  missing_specialization(b, :b_arc_dimension, c, r, α, Δα, rstr, dstr, size, offset, mat)

##################################################################
# Text
# To ensure a portable font, we will 'draw' the letters
const letter_glyph = Dict(
  ' '=>(bb=[(0, 0), (2/3, 1)], vss=[]),
  '!'=>(bb=[(0, 0), (0, 1)], vss=[[(0, 0), (0, 1/6)], [(0, 1/3), (0, 1)]]),
  '"'=>(bb=[(0, 2/3), (1/3, 1)], vss=[[(0, 2/3), (1/6, 1)], [(1/3, 1), (1/6, 2/3)]]),
  '#'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/3), (2/3, 1/3)], [(2/3, 2/3), (0, 2/3)], [(1/6, 1), (1/6, 0)], [(1/2, 0), (1/2, 1)]]),
  '$'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/6), (1/2, 1/6), (2/3, 1/3), (1/2, 1/2), (1/6, 1/2), (0, 2/3), (1/6, 5/6), (2/3, 5/6)], [(1/3, 1), (1/3, 0)]]),
  '%'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (0, 5/6), (1/6, 5/6), (1/6, 1), (0, 1)], [(2/3, 1), (0, 0)], [(2/3, 0), (1/2, 0), (1/2, 1/6), (2/3, 1/6), (2/3, 0)]]),
  '&'=>(bb=[(0, 0), (2/3, 1)], vss=[[(2/3, 1/3), (1/3, 0), (1/6, 0), (0, 1/6), (0, 1/3), (1/3, 2/3), (1/3, 5/6), (1/6, 1), (0, 5/6), (0, 2/3), (2/3, 0)]]),
  '\''=>(bb=[(0, 2/3), (1/6, 1)], vss=[[(0, 2/3), (1/6, 1)]]),
  '('=>(bb=[(0, 0), (1/3, 1)], vss=[[(1/3, 1), (0, 2/3), (0, 1/3), (1/3, 0)]]),
  ')'=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 1), (1/3, 2/3), (1/3, 1/3), (0, 0)]]),
  '*'=>(bb=[(0, 1/6), (2/3, 5/6)], vss=[[(1/3, 1/6), (1/3, 5/6)], [(2/3, 1/2), (0, 1/2)], [(2/3, 5/6), (0, 1/6)], [(0, 5/6), (2/3, 1/6)]]),
  '+'=>(bb=[(0, 1/6), (2/3, 5/6)], vss=[[(1/3, 1/6), (1/3, 5/6)], [(2/3, 1/2), (0, 1/2)]]),
  ','=>(bb=[(0, -1/6), (1/6, 1/6)], vss=[[(1/6, 1/6), (1/6, 0), (0, -1/6)]]),
  '-'=>(bb=[(0, 1/2), (2/3, 1/2)], vss=[[(0, 1/2), (2/3, 1/2)]]),
  '.'=>(bb=[(0, 0), (0, 1/6)], vss=[[(0, 0), (0, 1/6)]]),
  '/'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (2/3, 1)]]),
  '0'=>(bb=[(0, 0), (1/2, 1)], vss=[[(1/6, 0), (0, 1/6), (0, 5/6), (1/6, 1), (1/3, 1), (1/2, 5/6), (1/2, 1/6), (1/3, 0), (1/6, 0)]]),
  '1'=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 5/6), (1/6, 1), (1/6, 0)], [(0, 0), (1/3, 0)]]),
  '2'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 5/6), (1/6, 1), (1/2, 1), (2/3, 5/6), (2/3, 2/3), (1/2, 1/2), (1/6, 1/2), (0, 1/3), (0, 0), (2/3, 0)]]),
  '3'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 5/6), (1/6, 1), (1/2, 1), (2/3, 5/6), (2/3, 2/3), (1/2, 1/2), (1/3, 1/2)], [(1/2, 1/2), (2/3, 1/3), (2/3, 1/6), (1/2, 0), (1/6, 0), (0, 1/6)]]),
  '4'=>(bb=[(0, 0), (2/3, 1)], vss=[[(2/3, 1/3), (0, 1/3), (1/2, 1), (1/2, 0)]]),
  '5'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/6), (1/6, 0), (1/2, 0), (2/3, 1/6), (2/3, 1/2), (1/2, 2/3), (0, 2/3), (0, 1), (2/3, 1)]]),
  '6'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/2), (1/2, 1/2), (2/3, 1/3), (2/3, 1/6), (1/2, 0), (1/6, 0), (0, 1/6), (0, 2/3), (1/3, 1), (1/2, 1)]]),
  '7'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (2/3, 1), (1/6, 0)]]),
  '8'=>(bb=[(0, 0), (2/3, 1)], vss=[[(1/6, 0), (0, 1/6), (0, 1/3), (1/6, 1/2), (1/2, 1/2), (2/3, 2/3), (2/3, 5/6), (1/2, 1), (1/6, 1), (0, 5/6), (0, 2/3), (1/6, 1/2)], [(1/2, 1/2), (2/3, 1/3), (2/3, 1/6), (1/2, 0), (1/6, 0)]]),
  '9'=>(bb=[(0, 0), (2/3, 1)], vss=[[(1/6, 0), (1/3, 0), (2/3, 1/3), (2/3, 5/6), (1/2, 1), (1/6, 1), (0, 5/6), (0, 2/3), (1/6, 1/2), (2/3, 1/2)]]),
  ':'=>(bb=[(0, 1/6), (0, 2/3)], vss=[[(0, 2/3), (0, 1/2)], [(0, 1/3), (0, 1/6)]]),
  ';'=>(bb=[(0, -1/6), (1/6, 2/3)], vss=[[(1/6, 2/3), (1/6, 1/2)], [(1/6, 1/3), (1/6, 0), (0, -1/6)]]),
  '<'=>(bb=[(0, 0), (1/2, 1)], vss=[[(1/2, 1), (0, 1/2), (1/2, 0)]]),
  '='=>(bb=[(0, 1/3), (2/3, 2/3)], vss=[[(0, 2/3), (2/3, 2/3)], [(2/3, 1/3), (0, 1/3)]]),
  '>'=>(bb=[(0, 0), (1/2, 1)], vss=[[(0, 1), (1/2, 1/2), (0, 0)]]),
  '?'=>(bb=[(0, 0), (1/2, 1)], vss=[[(0, 5/6), (1/6, 1), (1/3, 1), (1/2, 5/6), (1/2, 2/3), (1/3, 1/2), (1/3, 1/3)], [(1/3, 1/6), (1/3, 0)]]),
  '@'=>(bb=[(0, 0), (2/3, 1)], vss=[[(1/2, 1/2), (1/3, 1/3), (1/6, 1/3), (1/6, 1/2), (1/3, 2/3), (1/2, 2/3), (1/2, 1/3), (2/3, 1/2), (2/3, 5/6), (1/2, 1), (1/6, 1), (0, 5/6), (0, 1/6), (1/6, 0), (2/3, 0)]]),
  'A'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1/3), (1/3, 1), (2/3, 1/3), (2/3, 0)], [(0, 1/3), (2/3, 1/3)]]),
  'B'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (1/2, 0), (2/3, 1/6), (2/3, 1/3), (1/2, 1/2), (1/6, 1/2)], [(1/2, 1/2), (2/3, 2/3), (2/3, 5/6), (1/2, 1), (0, 1)], [(1/6, 1), (1/6, 0)]]),
  'C'=>(bb=[(0, 0), (2/3, 1)], vss=[[(2/3, 1/6), (1/2, 0), (1/6, 0), (0, 1/6), (0, 5/6), (1/6, 1), (1/2, 1), (2/3, 5/6)]]),
  'D'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (1/2, 0), (2/3, 1/6), (2/3, 5/6), (1/2, 1), (0, 1)], [(1/6, 1), (1/6, 0)]]),
  'E'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (2/3, 1)], [(0, 1/2), (1/3, 1/2)], [(0, 0), (2/3, 0)]]),
  'F'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (2/3, 1)], [(0, 1/2), (1/3, 1/2)]]),
  'G'=>(bb=[(0, 0), (2/3, 1)], vss=[[(1/2, 1/2), (2/3, 1/2), (2/3, 0), (1/6, 0), (0, 1/6), (0, 5/6), (1/6, 1), (2/3, 1)]]),
  'H'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1)], [(0, 1/2), (2/3, 1/2)], [(2/3, 1), (2/3, 0)]]),
  'I'=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 1), (1/3, 1)], [(1/6, 1), (1/6, 0)], [(0, 0), (1/3, 0)]]),
  'J'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/6), (1/6, 0), (1/2, 0), (2/3, 1/6), (2/3, 1)]]),
  'K'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1)], [(2/3, 1), (1/6, 1/2), (0, 1/2)], [(1/6, 1/2), (2/3, 0)]]),
  'L'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (0, 0), (2/3, 0)]]),
  'M'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (1/3, 1/3), (2/3, 1), (2/3, 0)]]),
  'N'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (2/3, 0), (2/3, 1)]]),
  'O'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (2/3, 1), (2/3, 0), (0, 0)]]),
  'P'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (1/2, 1), (2/3, 5/6), (2/3, 2/3), (1/2, 1/2), (0, 1/2)]]),
  'Q'=>(bb=[(0, 0), (2/3, 1)], vss=[[(1/3, 1/3), (1/2, 1/6), (1/3, 0), (1/6, 0), (0, 1/6), (0, 5/6), (1/6, 1), (1/2, 1), (2/3, 5/6), (2/3, 1/3), (1/2, 1/6), (2/3, 0)]]),
  'R'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (1/2, 1), (2/3, 5/6), (2/3, 2/3), (1/2, 1/2), (0, 1/2)], [(1/6, 1/2), (2/3, 0)]]),
  'S'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/6), (1/6, 0), (1/2, 0), (2/3, 1/6), (0, 5/6), (1/6, 1), (1/2, 1), (2/3, 5/6)]]),
  'T'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (2/3, 1)], [(1/3, 1), (1/3, 0)]]),
  'U'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (0, 1/6), (1/6, 0), (1/2, 0), (2/3, 1/6), (2/3, 1)]]),
  'V'=>(bb=[(0, 0), (1, 1)], vss=[[(0, 1), (1/2, 0), (1, 1)]]),
  'W'=>(bb=[(0, 0), (1, 1)], vss=[[(0, 1), (1/3, 0), (1/2, 1/2), (2/3, 0), (1, 1)]]),
  'X'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (2/3, 1)], [(0, 1), (2/3, 0)]]),
  'Y'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (1/3, 1/2), (1/3, 0)], [(1/3, 1/2), (2/3, 1)]]),
  'Z'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (2/3, 1), (0, 0), (2/3, 0)]]),
  '['=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 0), (0, 1), (1/3, 1)], [(1/3, 0), (0, 0)]]),
  '\\'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (2/3, 0)]]),
  ']'=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 1), (1/3, 1), (1/3, 0), (0, 0)]]),
  '^'=>(bb=[(0, 2/3), (2/3, 1)], vss=[[(0, 2/3), (1/3, 1), (2/3, 2/3)]]),
  '_'=>(bb=[(0, -1/6), (2/3, -1/6)], vss=[[(0, -1/6), (2/3, -1/6)]]),
  '`'=>(bb=[(0, 2/3), (1/6, 1)], vss=[[(0, 1), (1/6, 2/3)]]),
  'a'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(1/3, 0), (1/6, 0), (0, 1/6), (0, 1/2), (1/6, 2/3), (1/3, 2/3), (1/2, 1/2), (1/2, 1/6), (1/3, 0)], [(1/2, 1/6), (2/3, 0)]]),
  'b'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1)], [(0, 1/3), (1/3, 2/3), (1/2, 2/3), (2/3, 1/2), (2/3, 1/6), (1/2, 0), (1/3, 0), (0, 1/3)]]),
  'c'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(2/3, 2/3), (1/6, 2/3), (0, 1/2), (0, 1/6), (1/6, 0), (2/3, 0)]]),
  'd'=>(bb=[(0, 0), (2/3, 1)], vss=[[(2/3, 1/3), (1/3, 0), (1/6, 0), (0, 1/6), (0, 1/2), (1/6, 2/3), (1/3, 2/3), (2/3, 1/3)], [(2/3, 1), (2/3, 0)]]),
  'e'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 1/3), (1/2, 1/3), (2/3, 1/2), (1/2, 2/3), (1/6, 2/3), (0, 1/2), (0, 1/6), (1/6, 0), (1/2, 0)]]),
  'f'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/2), (1/2, 1/2)], [(2/3, 5/6), (1/2, 1), (1/3, 1), (1/6, 5/6), (1/6, 0)]]),
  'g'=>(bb=[(0, -1/3), (2/3, 2/3)], vss=[[(0, -1/6), (1/6, -1/3), (1/2, -1/3), (2/3, -1/6), (2/3, 1/2), (1/2, 2/3), (1/6, 2/3), (0, 1/2), (0, 1/6), (1/6, 0), (2/3, 0)]]),
  'h'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1)], [(0, 1/3), (1/3, 2/3), (1/2, 2/3), (2/3, 1/2), (2/3, 0)]]),
  'i'=>(bb=[(0, 0), (0, 1)], vss=[[(0, 0), (0, 2/3)], [(0, 5/6), (0, 1)]]),
  'j'=>(bb=[(0, -1/3), (1/2, 1)], vss=[[(0, -1/6), (1/6, -1/3), (1/3, -1/3), (1/2, -1/6), (1/2, 2/3)], [(1/2, 5/6), (1/2, 1)]]),
  'k'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1)], [(0, 1/3), (1/3, 1/3), (2/3, 2/3)], [(1/3, 1/3), (2/3, 0)]]),
  'l'=>(bb=[(0, 0), (1/6, 1)], vss=[[(0, 1), (0, 1/6), (1/6, 0)]]),
  'm'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 0), (0, 2/3)], [(0, 1/2), (1/6, 2/3), (1/3, 1/2), (1/3, 1/3)], [(1/3, 1/2), (1/2, 2/3), (2/3, 1/2), (2/3, 0)]]),
  'n'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 0), (0, 2/3)], [(0, 1/3), (1/3, 2/3), (1/2, 2/3), (2/3, 1/2), (2/3, 0)]]),
  'o'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(1/2, 0), (1/6, 0), (0, 1/6), (0, 1/2), (1/6, 2/3), (1/2, 2/3), (2/3, 1/2), (2/3, 1/6), (1/2, 0)]]),
  'p'=>(bb=[(0, -1/3), (2/3, 2/3)], vss=[[(0, -1/3), (0, 2/3)], [(0, 1/2), (1/6, 2/3), (1/2, 2/3), (2/3, 1/2), (2/3, 1/6), (1/2, 0), (0, 0)]]),
  'q'=>(bb=[(0, -1/3), (2/3, 2/3)], vss=[[(2/3, -1/3), (2/3, 2/3)], [(2/3, 1/2), (1/2, 2/3), (1/6, 2/3), (0, 1/2), (0, 1/6), (1/6, 0), (2/3, 0)]]),
  'r'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 0), (0, 2/3)], [(0, 1/3), (1/3, 2/3), (1/2, 2/3), (2/3, 1/2)]]),
  's'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 0), (1/2, 0), (2/3, 1/6), (1/2, 1/3), (1/6, 1/3), (0, 1/2), (1/6, 2/3), (2/3, 2/3)]]),
  't'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 2/3), (2/3, 2/3)], [(1/3, 1), (1/3, 1/6), (1/2, 0), (2/3, 1/6)]]),
  'u'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 2/3), (0, 1/6), (1/6, 0), (1/3, 0), (2/3, 1/3)], [(2/3, 2/3), (2/3, 0)]]),
  'v'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 2/3), (1/3, 0), (2/3, 2/3)]]),
  'w'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 2/3), (1/6, 0), (1/3, 2/3), (1/2, 0), (2/3, 2/3)]]),
  'x'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 0), (2/3, 2/3)], [(0, 2/3), (2/3, 0)]]),
  'y'=>(bb=[(0, -1/3), (2/3, 2/3)], vss=[[(0, 2/3), (1/3, 0)], [(2/3, 2/3), (1/6, -1/3), (0, -1/3)]]),
  'z'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 2/3), (2/3, 2/3), (0, 0), (2/3, 0)]]),
  '{'=>(bb=[(0, 0), (1/3, 1)], vss=[[(1/3, 1), (1/6, 5/6), (1/6, 2/3), (0, 1/2), (1/6, 1/3), (1/6, 1/6), (1/3, 0)]]),
  '|'=>(bb=[(0, 0), (0, 1)], vss=[[(0, 0), (0, 1)]]),
  '}'=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 0), (1/6, 1/6), (1/6, 1/3), (1/3, 1/2), (1/6, 2/3), (1/6, 5/6), (0, 1)]]),
  '~'=>(bb=[(0, 1/2), (2/3, 2/3)], vss=[[(0, 1/2), (1/6, 2/3), (1/2, 1/2), (2/3, 2/3)]])
)

b_text(b::Backend, str, p, size, mat) =
  let dx = 0,
    inter_letter_spacing_factor = 1/3,
    refs = new_refs(b)
    for c in str
      let glyph = letter_glyph[c]
        for vs in glyph.vss
          push!(refs, b_line(b, [add_xy(p, dx + v[1]*size, v[2]*size) for v in vs], mat))
        end
        dx += (glyph.bb[2][1]-glyph.bb[1][1] + inter_letter_spacing_factor)*size
      end
    end
    refs
  end

b_text_size(b::Backend, str, size, mat) =
  let dx = 0, minx = 0, maxx = 0, miny = 0, maxy = 0,
    inter_letter_spacing_factor = 1/3
    for c in str
      let glyph = letter_glyph[c]
    minx = min(minx, dx + glyph.bb[1][1])
    maxx = max(maxx, dx + glyph.bb[2][1]-glyph.bb[1][1])
        dx += glyph.bb[2][1]-glyph.bb[1][1] + inter_letter_spacing_factor
    miny = min(miny, glyph.bb[1][2])
    maxy = max(maxy, glyph.bb[2][2])
      end
    end
  (minx, maxx, miny, maxy).*size
  end

##################################################################
# Illustrations

@bdef(b_labels(p, data, mat))
@bdef(b_radii_illustration(c, rs, rs_txts, mats, mat))
@bdef(b_vectors_illustration(p, a, rs, rs_txts, mats, mat))
@bdef(b_angles_illustration(c, rs, ss, as, r_txts, s_txts, a_txts, mats, mat))
@bdef(b_arcs_illustration(c, rs, ss, as, r_txts, s_txts, a_txts, mats, mat))

##################################################################
# Materials
#=
A material is a description of the appearence of an object.

There are two default cases:
1. When the specification of the material is nothing, we use the void_ref to
indicate that we don't want to specify the material.
2. Otherwise, the material already exists in the backend and is accessed by name.

In general, backends need to specialize this function to address additional cases.
=#

export b_get_material

b_get_material(b::Backend, spec::Nothing) = void_ref(b)
#Is this really needed? Yes, e.g., POVRay.
b_get_material(b::Backend, spec::Any) = spec

export BackendDefault, backend_default
struct BackendDefault end
const backend_default = BackendDefault()
b_get_material(b::Backend, ::BackendDefault) = void_ref(b)

#=
In other cases, the material can be algorithmically created.

There are numerous models for a material (Phong, Blinn–Phong, Cook–Torrance,
Lambert, Oren–Nayar, Minnaert, etc), covering wildly different materials such
as glass, metal, rubber, clay, etc. Unfortunately, different render engines
implement different subsets of those models, making it impossible to have portable
code that depends on a specific model. Additionally, each of these models have
different parameters, some of which might be difficult to specify by a non-expert.
Therefore, we prefer to provide a more intuitive approach, based on providing
different categories of materials, each with a minimal number of parameters.
Each backend is responsible for selecting the most adequate reflection model and
convert the generic material parameters into specific model parameters.

=#

# Material cascade: each tier strips parameters and calls the next tier down.
# Backends override at the tier matching their capabilities.

export b_material

# Tier 1 — Color only
b_material(b::Backend, name, base_color) =
  void_ref(b)

# Tier 2 — Basic PBR
b_material(b::Backend, name, base_color, metallic, roughness, specular) =
  b_material(b, name, base_color)

# Tier 3 — Standard PBR (transparency + coating + emission)
b_material(b::Backend, name, base_color, metallic, roughness, specular,
           ior, transmission, transmission_roughness,
           clearcoat, clearcoat_roughness,
           emission_color, emission_strength) =
  b_material(b, name, base_color, metallic, roughness, specular)

# Tier 4 — Full PBR (all Filament parameters)
b_material(b::Backend, name, base_color, metallic, roughness, specular,
           ior, transmission, transmission_roughness,
           clearcoat, clearcoat_roughness,
           emission_color, emission_strength,
           sheen_color, sheen_roughness,
           anisotropy, anisotropy_direction,
           ambient_occlusion, normal_map, bent_normal, clearcoat_normal,
           post_lighting_color,
           absorption, micro_thickness, thickness) =
  b_material(b, name, base_color, metallic, roughness, specular,
             ior, transmission, transmission_roughness,
             clearcoat, clearcoat_roughness,
             emission_color, emission_strength)

# Specialized material constructors — default to b_material with appropriate PBR values
export b_plastic_material, b_metal_material, b_glass_material, b_mirror_material

b_plastic_material(b::Backend, name, color, roughness) =
  b_material(b, name, color, 0.0, roughness, 0.5)

b_metal_material(b::Backend, name, color, roughness, ior) =
  b_material(b, name, color, 1.0, roughness, 0.9)

b_glass_material(b::Backend, name, color, roughness, ior) =
  b_material(b, name, color, 0.0, roughness, 0.5,
             ior, 0.8, roughness, 0.0, 0.0,
             rgba(0, 0, 0, 0), 0.0)

b_mirror_material(b::Backend, name, color) =
  b_material(b, name, color, 1.0, 0.0, 1.0)

#=
Backends might store shapes locally (in b.refs.shapes) or remotely
(in the CAD application). Some backends support querying all shapes
from the remote app (e.g., for processing existing DWG files).
=#
export ShapeStorageType, LocalShapeStorage, RemoteShapeStorage, shape_storage_type
abstract type ShapeStorageType end
struct LocalShapeStorage <: ShapeStorageType end
struct RemoteShapeStorage <: ShapeStorageType end

shape_storage_type(::Type{<:Backend}) = LocalShapeStorage()

# === Created shapes: always local, always from b.refs.shapes ===

export b_all_shape_refs, b_created_shape_refs, b_created_shapes,
       b_existing_shape_refs, b_existing_shapes

b_created_shape_refs(b::Backend) =
  collect(Iterators.flatten(ref_values(b, r) for r in values(b.refs.shapes)))

b_created_shapes(b::Backend) =
  collect(keys(b.refs.shapes))

# === Existing shapes: dispatch on trait ===

b_existing_shape_refs(b::T) where {T<:Backend} =
  b_existing_shape_refs(shape_storage_type(T), b)

b_existing_shape_refs(::LocalShapeStorage, b) =
  b_created_shape_refs(b)

b_existing_shape_refs(::RemoteShapeStorage, b) =
  missing_specialization(b, :b_existing_shape_refs)

b_existing_shapes(b::T) where {T<:Backend} =
  b_existing_shapes(shape_storage_type(T), b)

b_existing_shapes(::LocalShapeStorage, b) =
  b_created_shapes(b)

b_existing_shapes(::RemoteShapeStorage, b) =
  Shape[get_or_create_shape_from_ref_value(b, r) for r in b_existing_shape_refs(b)]

# === Combined: backward compatible ===

b_all_shape_refs(b::T) where {T<:Backend} =
  b_all_shape_refs(shape_storage_type(T), b)

b_all_shape_refs(::LocalShapeStorage, b) =
  b_created_shape_refs(b)

# For remote: the remote query already returns the superset (created + pre-existing)
b_all_shape_refs(::RemoteShapeStorage, b) =
  b_existing_shape_refs(b)

# === Deletion ===

b_delete_all_shape_refs(b::Backend) =
  b_delete_refs(b, b_all_shape_refs(b))

b_delete_refs(b::Backend{K,T}, rs::Vector{T}) where {K,T} =
  for r in rs
    b_delete_ref(b, r)
  end

b_delete_ref(b::Backend{K,T}, r::T) where {K,T} =
  missing_specialization(b, :b_delete_ref, r)

# === Highlight/Unhighlight ===

b_highlight_refs(b::Backend{K,T}, rs::Vector{T}) where {K,T} =
  for r in rs
    b_highlight_ref(b, r)
  end

b_highlight_ref(b::Backend{K,T}, r::T) where {K,T} =
  missing_specialization(b, :b_highlight_ref, r)

b_unhighlight_refs(b::Backend{K,T}, rs::Vector{T}) where {K,T} =
  for r in rs
    b_unhighlight_ref(b, r)
  end

b_unhighlight_ref(b::Backend{K,T}, r::T) where {K,T} =
  missing_specialization(b, :b_unhighlight_ref, r)

# Only unhighlight shapes Khepri created (not pre-existing shapes)
b_unhighlight_all_refs(b::Backend) =
  b_unhighlight_refs(b, b_created_shape_refs(b))

# BIM
export b_slab, b_roof, b_ceiling, b_beam, b_column, b_free_column, b_wall, b_curtain_wall,
       b_railing, b_ramp, b_stair, b_spiral_stair, b_stair_landing,
       b_wall_no_openings, b_wall_with_openings, WallOpening

#=
BIM operations require some extra support from the backends.
Given that humans prefer to live in horizontal surfaces, one interesting idea is
to separate 3D coordinates in two parts, one for the horizontal 2D coordinates
x and y, and another for the 1D vertical coordinate z, known as level.
=#

level_height(b::Backend, level) = level_height(level)

#=
Another relevant concept is the family. It contains generic information about
the construction element.
Given the wide variety of families (slabs, beams, windows, etc), it is not
possible to define a generic constructor or getter. However, for certain kinds
of families, it is possible to retrieve a set of materials that should be used
to represent the family. For a slab family, it should also be possible to
retrieve its thickness.
=#

#=
Horizontal BIM elements rely on the level
=#
export material_ref, material_refs
material_refs(b::Backend, materials) =
  [material_ref(b, mat) for mat in materials]

materialize_path(b, c_r_w_path, c_l_w_path, mat) =
  with_material_as_layer(b, mat) do
    b_strip(b, c_l_w_path, c_r_w_path, material_ref(b, mat))
  end
materialize_path(b, path, mat) =
  with_material_as_layer(b, mat) do
    b_surface(b, path, material_ref(b, mat))
  end

# b_slab(b::Backend, profile, level, family) =
#   let tmat = family.top_material,
#       bmat = family.bottom_material,
#       smat = family.side_material,
#       bprof = translate(profile, vz(level_height(b, level) + slab_family_elevation(b, family))),
#       tprof = translate(bprof, vz(slab_family_thickness(b, family)))
#     [materialize_path(b, tprof, tmat),
#      materialize_path(b, bprof, tprof, smat)...,
#      materialize_path(b, reverse(bprof), bmat)]
#   end
# It is preferable to do an extrusion because separate discretization of the top and bottom
# paths do not ensure an equal number of vertices.
b_slab(b::Backend, profile, level, family) =
  let tmat = material_ref(b, family.top_material),
      bmat = material_ref(b, family.bottom_material),
      smat = material_ref(b, family.side_material),
      v = vz(slab_family_thickness(b, family))
    b_extruded_surface(b, profile, v, z(level_height(b, level) + slab_family_elevation(b, family)), bmat, tmat, smat)
   end

b_roof(b::Backend, region, level, family) =
  b_slab(b, region, level, family)

b_ceiling(b::Backend, profile, level, family) =
  let tmat = material_ref(b, family.top_material),
      bmat = material_ref(b, family.bottom_material),
      smat = material_ref(b, family.side_material),
      th = ceiling_family_thickness(b, family)
    b_extruded_surface(b, profile, vz(th),
      z(level_height(b, level) - th + ceiling_family_elevation(b, family)),
      bmat, tmat, smat)
  end

b_railing(b::Backend, path, level, host, family) =
  let h = family.height,
      base = level_height(b, level),
      mat = material_ref(b, family.material),
      len = path_length(path),
      n_posts = max(2, Int(ceil(len / family.post_spacing)) + 1),
      post_locs = [location_at_length(path, t)
                   for t in division(0, len, n_posts - 1)]
    with_material_as_layer(b, family.material) do
      vcat(
        b_sweep(b, translate(path, vz(base + h)),
                rectangular_profile(0.05, 0.05), 0, 1, mat),
        [b_cylinder(b, add_z(in_world(pt), base), 0.025, h, mat, mat, mat)
         for pt in post_locs]...)
    end
  end

b_ramp(b::Backend, path, bottom_level, top_level, family) =
  let bottom_h = level_height(b, bottom_level),
      top_h = level_height(b, top_level),
      w = family.width,
      th = family.thickness,
      tmat = material_ref(b, family.top_material),
      bmat = material_ref(b, family.bottom_material),
      smat = material_ref(b, family.side_material),
      p0 = in_world(path_start(path)),
      p1 = in_world(path_end(path)),
      dir = unitized(p1 - p0),
      perp = cross(dir, vz(1))
    b_pyramid_frustum(b,
      [p0 + perp*(w/2) + vz(bottom_h),
       p0 - perp*(w/2) + vz(bottom_h),
       p0 - perp*(w/2) + vz(bottom_h - th),
       p0 + perp*(w/2) + vz(bottom_h - th)],
      [p1 + perp*(w/2) + vz(top_h),
       p1 - perp*(w/2) + vz(top_h),
       p1 - perp*(w/2) + vz(top_h - th),
       p1 + perp*(w/2) + vz(top_h - th)],
      bmat, tmat, smat)
  end

b_stair(b::Backend, base_point, direction, bottom_level, top_level, family) =
  let bottom_h = level_height(b, bottom_level),
      top_h = level_height(b, top_level),
      total_h = top_h - bottom_h,
      n_steps = Int(round(total_h / family.riser_height)),
      riser_h = total_h / n_steps,
      tread_d = family.tread_depth,
      w = family.width,
      dir = unitized(direction),
      perp = cross(vz(1), dir),
      tmat = material_ref(b, family.tread_material),
      rmat = material_ref(b, family.riser_material),
      refs = new_refs(b)
    for i in 0:(n_steps - 1)
      let base = in_world(base_point) + dir * (i * tread_d) + vz(bottom_h + i * riser_h),
          tread = [base + vz(riser_h),
                   base + perp * w + vz(riser_h),
                   base + dir * tread_d + perp * w + vz(riser_h),
                   base + dir * tread_d + vz(riser_h)]
        collect_ref!(refs, b_surface_polygon(b, tread, tmat))
        if family.has_risers
          riser = [base, base + perp * w,
                   base + perp * w + vz(riser_h), base + vz(riser_h)]
          collect_ref!(refs, b_surface_polygon(b, riser, rmat))
        end
      end
    end
    refs
  end

b_spiral_stair(b::Backend, center, radius, start_angle, included_angle,
               clockwise, bottom_level, top_level, family) =
  let bottom_h = level_height(b, bottom_level),
      top_h = level_height(b, top_level),
      total_h = top_h - bottom_h,
      n_steps = Int(round(total_h / family.riser_height)),
      riser_h = total_h / n_steps,
      sign = clockwise ? -1 : 1,
      angle_step = sign * included_angle / n_steps,
      w = family.width,
      r_inner = radius - w/2,
      r_outer = radius + w/2,
      tmat = material_ref(b, family.tread_material),
      c = in_world(center),
      refs = new_refs(b)
    for i in 0:(n_steps - 1)
      let a0 = start_angle + i * angle_step,
          a1 = a0 + angle_step,
          h = bottom_h + (i + 1) * riser_h,
          tread = [c + vpol(r_inner, a0) + vz(h),
                   c + vpol(r_outer, a0) + vz(h),
                   c + vpol(r_outer, a1) + vz(h),
                   c + vpol(r_inner, a1) + vz(h)]
        collect_ref!(refs, b_surface_polygon(b, tread, tmat))
      end
    end
    refs
  end

b_stair_landing(b::Backend, region, level, family) =
  b_slab(b, region, level, family)

# b_panel(b::Backend, region, family) =
#   let th = family.thickness,
#       v = planar_path_normal(region),
#       left = translate(region, v*-th),
#       right = translate(region, v*th)
#     [materialize_path(b, right, family.right_material),
#      materialize_path(b, left, right, family.side_material)...,
#      materialize_path(b, reverse(left), family.left_material)]
#   end
b_panel(b::Backend, profile, family) =
  let lmat = material_ref(b, family.left_material),
      rmat = material_ref(b, family.right_material),
      smat = material_ref(b, family.side_material),
      th = family.thickness,
      v = planar_path_normal(profile)
    b_extruded_surface(b, profile, v*th, u0(v.cs)+v*(th/-2), lmat, rmat, smat)
  end

b_beam(b::Backend, c, h, angle, family) =
  let c = loc_from_o_phi(c, angle),
      mat = material_ref(b, family.material)
    with_material_as_layer(b, family.material) do
      b_extruded_surface(b, region(family_profile(b, family)), vz(h, c.cs), c,  mat, mat, mat)
    end
  end

b_column(b::Backend, cb, angle, bottom_level, top_level, family) =
  let base_height = level_height(b, bottom_level),
      top_height = level_height(b, top_level)
    b_beam(b, add_z(loc_from_o_phi(cb, angle), base_height), top_height-base_height, 0, family)
  end

b_free_column(b::Backend, cb, h, angle, family) =
  b_beam(b, cb, h, angle, family)

struct WallOpening
  path_position::Real
  base_height::Real
  width::Real
  height::Real
end

# Whole-path wall geometry: 4 quad strips + 2 end caps (open paths) = 4-6 calls total
b_wall_no_openings(b::Backend, w_path, w_height, l_thickness, r_thickness, lmat, rmat, smat) =
  path_length(w_path) < path_tolerance() ?
    void_ref(b) :
    let r_path = offset(w_path, -r_thickness),
        l_path = offset(w_path, l_thickness),
        w_height = w_height * wall_z_fighting_factor,
        r_vs = path_vertices(r_path),
        l_vs = path_vertices(l_path),
        r_top = map(p -> p + vz(w_height), r_vs),
        l_top = map(p -> p + vz(w_height), l_vs),
        closed = is_closed_path(w_path),
        strip = closed ? b_quad_strip_closed : b_quad_strip,
        refs = new_refs(b)
      with_material_as_layer(b, rmat) do
        collect_ref!(refs, strip(b, r_vs, r_top, false, material_ref(b, rmat)))
      end
      with_material_as_layer(b, lmat) do
        collect_ref!(refs, strip(b, l_top, l_vs, false, material_ref(b, lmat)))
      end
      with_material_as_layer(b, smat) do
        let smat_ref = material_ref(b, smat)
          collect_ref!(refs, strip(b, r_top, l_top, false, smat_ref))
          collect_ref!(refs, strip(b, l_vs, r_vs, false, smat_ref))
          if !closed
            collect_ref!(refs, b_surface_polygon(b, [r_vs[1], r_top[1], l_top[1], l_vs[1]], smat_ref))
            collect_ref!(refs, b_surface_polygon(b, [l_vs[end], l_top[end], r_top[end], r_vs[end]], smat_ref))
          end
        end
      end
      refs
    end

b_wall_no_openings(b::Backend, w_path, w_height, l_thickness, r_thickness, family) =
  b_wall_no_openings(b, w_path, w_height, l_thickness, r_thickness,
                     family.left_material, family.right_material, family.side_material)

# Per-segment wall geometry with inline hole punching for backends without boolean ops
subtract_wall_paths(b::Backend, c_r_w_path, c_l_w_path, c_r_op_path, c_l_op_path) =
  region(c_r_w_path, c_r_op_path), region(c_l_w_path, c_l_op_path)

b_wall_with_openings(b::Backend, w_path, w_height, l_thickness, r_thickness, lmat, rmat, smat, openings) =
  path_length(w_path) < path_tolerance() ?
    void_ref(b) :
    let w_paths = subpaths(w_path),
        r_w_paths = subpaths(offset(w_path, -r_thickness)),
        l_w_paths = subpaths(offset(w_path, l_thickness)),
        w_height = w_height * wall_z_fighting_factor,
        prevlength = 0,
        refs = new_refs(b)
      for (w_seg_path, r_w_path, l_w_path) in zip(w_paths, r_w_paths, l_w_paths)
        let currlength = prevlength + path_length(w_seg_path),
            c_r_w_path = closed_path_for_height(r_w_path, w_height),
            c_l_w_path = closed_path_for_height(l_w_path, w_height)
          # Render top edge and end caps only (omit bottom face to avoid
          # sliver across door openings at floor level)
          with_material_as_layer(b, smat) do
            let smat_ref = material_ref(b, smat),
                r_vs = path_vertices(r_w_path),
                l_vs = path_vertices(l_w_path),
                r_top = map(p -> p + vz(w_height), r_vs),
                l_top = map(p -> p + vz(w_height), l_vs)
              collect_ref!(refs, b_quad_strip(b, r_top, l_top, false, smat_ref))
              if !is_closed_path(w_path)
                collect_ref!(refs, b_surface_polygon(b, [r_vs[1], r_top[1], l_top[1], l_vs[1]], smat_ref))
                collect_ref!(refs, b_surface_polygon(b, [l_vs[end], l_top[end], r_top[end], r_vs[end]], smat_ref))
              end
            end
          end
          openings = filter(openings) do op
            if prevlength <= op.path_position < currlength ||
               prevlength <= op.path_position + op.width <= currlength
              let op_height = op.height,
                  op_at_start = op.path_position <= prevlength,
                  op_at_end = op.path_position + op.width >= currlength,
                  op_path = subpath(w_path,
                                    max(prevlength, op.path_position),
                                    min(currlength, op.path_position + op.width)),
                  r_op_path = offset(op_path, -r_thickness),
                  l_op_path = offset(op_path,  l_thickness),
                  fixed_r_op_path =
                    open_polygonal_path([path_start(op_at_start ? r_w_path : r_op_path),
                                         path_end(op_at_end ? r_w_path : r_op_path)]),
                  fixed_l_op_path =
                    open_polygonal_path([path_start(op_at_start ? l_w_path : l_op_path),
                                         path_end(op_at_end ? l_w_path : l_op_path)]),
                  r_op_translated = translate(fixed_r_op_path, vz(op.base_height)),
                  l_op_translated = translate(fixed_l_op_path, vz(op.base_height)),
                  c_r_op_path = closed_path_for_height(r_op_translated, op_height),
                  c_l_op_path = closed_path_for_height(l_op_translated, op_height),
                  # Jacket: open U-shape (no sill) for floor-level openings, full closed for elevated
                  r_jacket = op.base_height < path_tolerance() ?
                    let ps = path_vertices(r_op_translated)
                      open_polygonal_path([ps[1], ps[1]+vz(op_height), ps[end]+vz(op_height), ps[end]])
                    end : c_r_op_path,
                  l_jacket = op.base_height < path_tolerance() ?
                    let ps = path_vertices(l_op_translated)
                      open_polygonal_path([ps[1], ps[1]+vz(op_height), ps[end]+vz(op_height), ps[end]])
                    end : c_l_op_path
                collect_ref!(refs, materialize_path(b, reverse(r_jacket), reverse(l_jacket), smat))
                c_r_w_path, c_l_w_path = subtract_wall_paths(b, c_r_w_path, c_l_w_path, c_r_op_path, c_l_op_path)
                !(op.path_position >= prevlength && op.path_position + op.width <= currlength)
              end
            else
              true
            end
          end
          prevlength = currlength
          # Render wall faces (with openings subtracted as holes in the surfaces)
          collect_ref!(refs, materialize_path(b, reverse(c_l_w_path), lmat))
          collect_ref!(refs, materialize_path(b, c_r_w_path, rmat))
        end
      end
      refs
    end

# Main b_wall dispatcher: family + offset → decompose and dispatch
b_wall(b::Backend, w_path, w_height, family, offset, openings) =
  let l_th = (1/2 + offset) * (family.thickness + family.left_coating_thickness),
      r_th = (1/2 - offset) * (family.thickness + family.right_coating_thickness),
      lmat = family.left_material,
      rmat = family.right_material,
      smat = family.side_material
    isempty(openings) ?
      b_wall_no_openings(b, w_path, w_height, l_th, r_th, lmat, rmat, smat) :
      b_wall_with_openings(b, w_path, w_height, l_th, r_th, lmat, rmat, smat, openings)
  end

b_curtain_wall(b::Backend, path, bottom_level, top_level, family, offset) =
  let th = family.panel.thickness,
      bfw = family.boundary_frame.width,
      bfd = family.boundary_frame.depth,
      bfdo = family.boundary_frame.depth_offset,
      mfw = family.mullion_frame.width,
      mfd = family.mullion_frame.depth,
      mdfo = family.mullion_frame.depth_offset,
      tfw = family.transom_frame.width,
      tfd = family.transom_frame.depth,
      tfdo = family.transom_frame.depth_offset,
      path = curtain_wall_panel_path(b, path, family),
      path_length = path_length(path),
      bottom = level_height(b, bottom_level),
      top = level_height(b, top_level),
      height = top - bottom,
      x_panels = ceil(Int, path_length/family.max_panel_dx),
      y_panels = ceil(Int, height/family.max_panel_dy),
      refs = new_refs(b)
    collect_ref!(refs, b_curtain_wall_element(b, subpath(path, bfw, path_length-bfw), bottom+bfw, height-2*bfw, th/2, th/2, getproperty(family, :panel)))
    collect_ref!(refs, b_curtain_wall_element(b, path, bottom, bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), getproperty(family, :boundary_frame)))
    collect_ref!(refs, b_curtain_wall_element(b, path, top-bfw, bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), getproperty(family, :boundary_frame)))
    collect_ref!(refs, b_curtain_wall_element(b, subpath(path, 0, bfw), bottom+bfw, height-2*bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), getproperty(family, :boundary_frame)))
    collect_ref!(refs, b_curtain_wall_element(b, subpath(path, path_length-bfw, path_length), bottom+bfw, height-2*bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), getproperty(family, :boundary_frame)))
    for i in 1:y_panels-1
      l = height/y_panels*i
      sub = subpath(path, bfw, path_length-bfw)
      collect_ref!(refs, b_curtain_wall_element(b, sub, bottom+l-tfw/2, tfw, l_thickness(tfdo, tfd), r_thickness(tfdo, tfd), getproperty(family, :transom_frame)))
    end
    for i in 1:x_panels-1
      l = path_length/x_panels*i
      collect_ref!(refs, b_curtain_wall_element(b, subpath(path, l-mfw/2, l+mfw/2), bottom+bfw, height-2*bfw, l_thickness(mdfo, mfd), r_thickness(mdfo, mfd), getproperty(family, :mullion_frame)))
    end
    refs
  end

curtain_wall_panel_path(b::Backend, path, family) =
  let path_length = path_length(path),
      x_panels = ceil(Int, path_length/family.max_panel_dx),
      pts = map(t->in_world(location_at_length(path, t)),
                division(0, path_length, x_panels))
    polygonal_path(pts)
  end

## ─────────────────────────────────────────────────────────────────────
##  OBJ/MTL family types and transform computation
##
##  OBJ family types must be defined here (before BIM.jl) because
##  the transform functions and BIM default fallbacks below use them.
##  Used by any backend that implements b_mesh_obj_fmt.
## ─────────────────────────────────────────────────────────────────────

abstract type OBJFamily <: Family end

struct OBJFileFamily <: OBJFamily
  obj_name::String    # relative subpath under resources/models/obj/ (without .obj)
  scale::Float64
  rotation::Float64   # rotation around vertical axis (radians)
  offset::Vec         # local offset in model coordinates
  y_is_up::Bool       # true if OBJ uses Y-up convention (default false = Z-up)
end

export OBJFamily, OBJFileFamily, obj_family

# obj_name is a relative subpath under resources/models/obj/:
#   obj_family("name/name")  → subfolder layout (name/name.obj)
#   obj_family("name")       → flat layout (name.obj)
obj_family(obj_name; scale=1.0, rotation=0.0, offset=vxyz(0, 0, 0), y_is_up=false) =
  OBJFileFamily(obj_name, Float64(scale), Float64(rotation), offset, y_is_up)

# OBJ families are backend-level families (the value in implemented_as).
# backend_get_family_ref returns the family itself — actual loading
# happens in b_mesh_obj_fmt at element placement time.
backend_get_family_ref(b::Backend, f::Family, bf::OBJFileFamily) = bf

export standalone_obj_transform, wall_obj_transform

#=
  standalone_obj_transform(position, bf::OBJFileFamily)

  Compute a Loc (4×4 transform) for placing an OBJ mesh at a world
  position (no wall context). Rotation is around the world Z (vertical) axis.
=#
function standalone_obj_transform(position, bf::OBJFileFamily)
  let p = in_world(position),
      s = bf.scale,
      θ = bf.rotation,
      cθ = cos(θ), sθ = sin(θ)
    if bf.y_is_up
      # Y-up OBJ → Z-up world: X→X, Y→Z (up), Z→-Y (preserves handedness)
      let vx = vxyz(cθ * s, sθ * s, 0),
          vy = vxyz(0, 0, s),
          vz = vxyz(sθ * s, -cθ * s, 0),
          local_offset = vx * bf.offset.x + vy * bf.offset.y + vz * bf.offset.z
        u0(cs_from_o_vx_vy_vz(p + local_offset, vx, vy, vz))
      end
    else
      # Z-up OBJ (default): identity axis mapping
      let vx = vxyz(cθ * s, sθ * s, 0),
          vy = vxyz(-sθ * s, cθ * s, 0),
          vz = vxyz(0, 0, s),
          local_offset = vx * bf.offset.x + vy * bf.offset.y + vz * bf.offset.z
        u0(cs_from_o_vx_vy_vz(p + local_offset, vx, vy, vz))
      end
    end
  end
end

#=
  wall_obj_transform(sp_begin, sp_end, base_height, bf::OBJFileFamily)

  Compute a Loc (4×4 transform) to position and orient an OBJ mesh
  at a door/window opening on a wall.

  The resulting coordinate system:
    X = along the wall (tangent), rotated by bf.rotation
    Y = vertical (up) or wall normal, depending on y_is_up
    Z = wall normal or vertical (up)
=#
function wall_obj_transform(sp_begin, sp_end, base_height, bf::OBJFileFamily)
  let p = in_world(sp_begin) + vz(base_height),
      tangent = unitized(in_world(sp_end) - in_world(sp_begin)),
      up = vz(1),
      normal = unitized(cross(tangent, up)),
      s = bf.scale,
      θ = bf.rotation,
      cθ = cos(θ), sθ = sin(θ),
      rt = tangent * cθ + normal * sθ,
      rn = -tangent * sθ + normal * cθ
    if bf.y_is_up
      # Y-up OBJ → Z-up world: X→tangent, Y→up, Z→-normal (preserves handedness)
      let vx = rt * s,
          vy = up * s,
          vz = -rn * s,
          local_offset = rt * bf.offset.x + up * bf.offset.y - rn * bf.offset.z
        u0(cs_from_o_vx_vy_vz(p + local_offset, vx, vy, vz))
      end
    else
      # Z-up OBJ (default): model X → tangent, model Y → normal, model Z → up
      let vx = rt * s,
          vy = rn * s,
          vz = up * s,
          local_offset = rt * bf.offset.x + rn * bf.offset.y + up * bf.offset.z
        u0(cs_from_o_vx_vy_vz(p + local_offset, vx, vy, vz))
      end
    end
  end
end

## ─────────────────────────────────────────────────────────────────────
##  OBJ-based placement helpers
## ─────────────────────────────────────────────────────────────────────

export place_obj, place_obj_oriented, place_obj_at_wall

place_obj(bf::OBJFileFamily, p::Loc=u0()) =
  b_mesh_obj_fmt(current_backend(), bf.obj_name, standalone_obj_transform(p, bf))

function place_obj_oriented(bf::OBJFileFamily, p::Loc, dir::Vec)
  let pw = in_world(p),
      tangent = unitized(in_world(dir)),
      up = vz(1),
      normal = unitized(cross(tangent, up)),
      s = bf.scale,
      local_offset = tangent * bf.offset.x + up * bf.offset.y + normal * bf.offset.z,
      vx = tangent * s,
      vy = up * s,
      vz_out = normal * s
    b_mesh_obj_fmt(current_backend(), bf.obj_name,
      u0(cs_from_o_vx_vy_vz(pw + local_offset, vx, vy, vz_out)))
  end
end

function place_obj_at_wall(bf::OBJFileFamily, w, dist, height; side=1)
  let sp = subpath(w.path, dist, dist + 0.01),
      tangent = unitized(in_world(sp[end]) - in_world(sp[begin])),
      up = vz(1),
      normal = unitized(cross(tangent, up)) * side,
      p = in_world(sp[begin]) + vz(w.bottom_level.height + height),
      s = bf.scale,
      θ = bf.rotation,
      cθ = cos(θ), sθ = sin(θ),
      rt = tangent * cθ + normal * sθ,
      rn = -tangent * sθ + normal * cθ,
      local_offset = rt * bf.offset.x + up * bf.offset.y + rn * bf.offset.z,
      vx = rt * s,
      vy = up * s,
      vz_out = rn * s
    b_mesh_obj_fmt(current_backend(), bf.obj_name,
      u0(cs_from_o_vx_vy_vz(p + local_offset, vx, vy, vz_out)))
  end
end

## ─────────────────────────────────────────────────────────────────────
##  BIM fixture defaults (toilet, sink, closet)
##
##  When the family has an OBJFamily backend implementation, the OBJ
##  model is loaded at the element's position. Otherwise, a simple
##  box placeholder is used.
## ─────────────────────────────────────────────────────────────────────

export b_toilet, b_sink, b_closet
b_toilet(b::Backend, c, host, family) =
  let bf = get(family.implemented_as, typeof(b), nothing)
    bf isa OBJFamily ?
      b_mesh_obj_fmt(b, bf.obj_name, standalone_obj_transform(c, bf)) :
      b_box(b, c - vxy(20, 20, c.cs), 40, 40, 40, nothing)
  end

b_sink(b::Backend, c, host, family) =
  let bf = get(family.implemented_as, typeof(b), nothing)
    bf isa OBJFamily ?
      b_mesh_obj_fmt(b, bf.obj_name, standalone_obj_transform(c, bf)) :
      b_box(b, c - vxy(40, 40, c.cs), 80, 80, 80, nothing)
  end

b_closet(b::Backend, c, host, family) =
  let bf = get(family.implemented_as, typeof(b), nothing)
    bf isa OBJFamily ?
      b_mesh_obj_fmt(b, bf.obj_name, standalone_obj_transform(c, bf)) :
      b_box(b, c - vxy(100, 40, c.cs), 200, 80, 200, nothing)
  end

## ─────────────────────────────────────────────────────────────────────
##  OBJ file utilities
##
##  Path resolution and parsing for backends that load OBJ meshes by
##  reading vertex/face data on the Julia side (e.g., when the backend
##  has no native OBJ import).
## ─────────────────────────────────────────────────────────────────────

export obj_file_path, read_obj_mesh, transform_obj_vertices

#=
  obj_file_path(obj_name)

  Resolve an OBJ model subpath to its file path. The obj_name is a relative
  subpath under resources/models/obj/:
    obj_family("Porta/Porta")  → resources/models/obj/Porta/Porta.obj  (subfolder)
    obj_family("Porta")        → resources/models/obj/Porta.obj        (flat)
=#
obj_file_path(obj_name) =
  joinpath("resources", "models", "obj", "$obj_name.obj")

#=
  read_obj_mesh(filepath)

  Parse an OBJ file and return (vertices, faces) where:
    vertices — Vector of [x, y, z] Float64 arrays
    faces    — Vector of Int arrays (1-based vertex indices)

  Only processes 'v' (vertex) and 'f' (face) lines.
  Face indices with texture/normal components (v/vt/vn) are handled.
=#
function read_obj_mesh(filepath)
  vertices = Vector{Float64}[]
  faces = Vector{Int}[]
  open(filepath) do io
    for line in eachline(io)
      parts = split(strip(line))
      isempty(parts) && continue
      if parts[1] == "v" && length(parts) >= 4
        push!(vertices, [parse(Float64, parts[2]),
                         parse(Float64, parts[3]),
                         parse(Float64, parts[4])])
      elseif parts[1] == "f"
        face = Int[]
        for i in 2:length(parts)
          push!(face, parse(Int, split(parts[i], '/')[1]))
        end
        push!(faces, face)
      end
    end
  end
  (vertices, faces)
end

#=
  transform_obj_vertices(verts, transform)

  Apply a Loc transform (from standalone_obj_transform or wall_obj_transform)
  to raw OBJ vertex positions. Returns world-space Loc points.
=#
transform_obj_vertices(verts, transform) =
  [in_world(transform + vxyz(v[1], v[2], v[3])) for v in verts]

export b_family_element
b_family_element(b::Backend, loc, angle, level, family) =
  b_box(b, loc - vxy(0.5, 0.5, loc.cs), 1.0, 1.0, 1.0, nothing)

# Lights

@bdef b_pointlight(loc, energy, color)
@bdef b_spotlight(loc, dir, hotspot, falloff)

# Default fallbacks: approximate unsupported light types with simpler ones
export b_ieslight
b_ieslight(b::Backend, file, loc, dir, alpha, beta, gamma) =
  b_spotlight(b, loc, dir, pi/4, pi/3)

export b_arealight
b_arealight(b::Backend, loc, dir, size, energy, color) =
  b_pointlight(b, loc, energy, color)

# Trusses
export b_truss_node, b_truss_node_support, b_truss_bar

b_truss_node(b::Backend, p, family) =
  with_material_as_layer(b, family.material) do
    b_sphere(b, p, family.radius, material_ref(b, family.material))
  end

b_truss_node_support(b::Backend, cb, family) =
  with_material_as_layer(b, family.material) do
    b_regular_pyramid(
      b, 4, add_z(cb, -3*family.radius),
      family.radius, 0, 3*family.radius, false,
      material_ref(b, family.material))
  end

b_truss_bar(b::Backend, p, q, family) =
  with_material_as_layer(b, family.material) do
    let (c, h) = position_and_height(p, q)
      b_cylinder(
        b, c, family.radius, h,
        material_ref(b, family.material))
    end
  end

# Analysis

@bdef b_truss_analysis(load::Vec, self_weight::Bool, point_loads::Dict)
@bdef b_node_displacement_function(res::Any)

export b_truss_bars_volume
b_truss_bars_volume(b::Backend) =
  sum(truss_bar_volume, b.truss_bars)

#=
Operations that rely on a backend need to have a backend selected and will
generate an exception if there is none.
=#

struct UndefinedBackendException <: Exception end
showerror(io::IO, e::UndefinedBackendException) =
  print(io,
    "No backend is set. Call backend(my_backend) to select one.\n",
    "Example: using KhepriAutoCAD; backend(autocad)")

#=
Moving Khepri to a multi-threaded model requires that each thread uses its own
current backend.

This means that current_backends should only be used to intentionally broadcast
an operation to several backends at the same time. In all other cases, current_backend
should be used to retrieve the backend for the current thread. 

TO BE CONTINUED!
=#

# We can have several backends active at the same time
const Backends = Tuple{Vararg{Backend}}
const current_backends = Parameter{Backends}(())
has_current_backend() = !isempty(current_backends())

export add_current_backend, delete_current_backend
add_current_backend(b::Backend) =
  current_backends(tuple(b, current_backends()...))
delete_current_backend(b::Backend) =
  current_backends(filter(!=(b), current_backends()))

# Global variants: update the shared default so that tasks without
# a task-local override (e.g. the user's main REPL task) can see
# backends that connected via the socket/websocket server.
const global_backends_lock = ReentrantLock()
export add_global_backend, delete_global_backend
add_global_backend(b::Backend) =
  lock(global_backends_lock) do
    current_backends.value = tuple(b, current_backends.value...)
  end
delete_global_backend(b::Backend) =
  lock(global_backends_lock) do
    current_backends.value = Tuple(filter(!=(b), collect(current_backends.value)))
  end
 
# but for backward compatibility reasons, we might also select just one.
top_backend() =
  let bs = current_backends(),
      i = 0
    while isempty(bs) && i < 10
      @info("Waiting for a backend to be available...")
      sleep(5)
      bs = current_backends()
      i += 1
    end
    isempty(bs) ? throw(UndefinedBackendException()) : bs[1]
  end

# The current_backend function is just an alias for current_backends
current_backend() = current_backends()
current_backend(bs::Backends) = current_backends(bs)
# but it knows how to treat a single backend.
current_backend(b::Backend) = current_backends((b,))

backend(backend::Backend) =
  has_current_backend() ?
    switch_to_backend(top_backend(), backend) :
    current_backend(backend)
switch_to_backend(from::Backend, to::Backend) =
  current_backend(to)

export top_backend
export purge_backends
purge_backends() =
  let bs = current_backends(),
      ok_bs = []
    for b in bs
      try
        with(current_backend, b) do
          get_view() # use a more efficient operation
        end
        push!(ok_bs, b)
      catch e
        @info("Backend $(b.name) is dead!")
      end
    end
    current_backends((ok_bs...,))
  end

# Variables with backend-specific values can be useful.
# Basically, they are dictionaries.
# but they also support a default value for the case
# where there is no backend-specific value available
export BackendParameter
struct BackendParameter
  value::IdDict{Type{<:Backend}, Any}
  default::Any
  BackendParameter(ps...; default=nothing) = new(IdDict{Type{<:Backend}, Any}(ps...), default)
  BackendParameter(p::BackendParameter) = new(copy(p.value), p.default)
end

(p::BackendParameter)(b::Backend=top_backend()) = error("Don't do this") #get(p.value, b, nothing)
(p::BackendParameter)(b::Backend, newvalue) = error("Don't do this") #p.value[b] = newvalue

(p::BackendParameter)(tb::Type{<:Backend}) = get(p.value, tb, p.default)
(p::BackendParameter)(tb::Type{<:Backend}, newvalue) = p.value[tb] = newvalue


Base.copy(p::BackendParameter) = BackendParameter(p)

export @backend
macro backend(b, expr)
  quote
  with(current_backends, ($(esc(b)),)) do
    $(esc(expr))
    end
  end
end

export @backends
macro backends(b, expr)
  quote
  with(current_backends, $(esc(b))) do
    $(esc(expr))
    end
  end
end

## THIS NEEDS TO BE SIMPLIFIED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Many functions default the backend to the top_backend and throw an error if there is none.
# We will simplify their definition with a macro:
# @defop delete_all_shapes()
# that expands into
# delete_all_shapes(backend::Backend=top_backend()) = throw(UndefinedBackendException())
# Note that according to Julia semantics the previous definition actually generates two different ones:
# delete_all_shapes() = delete_all_shapes(top_backend())
# delete_all_shapes(backend::Backend) = throw(UndefinedBackendException())
# Hopefully, backends will specialize the function for each specific backend


#=

Each shape can have a material and a classification.
A material might be just a color or something more complex
a classification might be a layer in some CAD tools or something else
Some backends will ignore this information, some use it in a dynamic
fashion. For performance reasons, operations that create shapes can
also be used without materials or classification.

=#

#@bdef add_door(w::Wall, loc::Loc, family::DoorFamily)
#@bdef add_window(w::Wall, loc::Loc, family::WindowFamily)
#@bdef bounding_box(shapes::Shapes)

#@bdef chair(c, angle, family)
export b_table
b_table(b::Backend, p, length, width, height, top_thickness, leg_thickness, mat) =
  let dx = length/2,
    dy = width/2,
    leg_x = dx - leg_thickness/2,
    leg_y = dy - leg_thickness/2,
    c = add_xy(p, -dx, -dy),
    table_top = b_box(b, add_z(c, height - top_thickness), length, width, top_thickness, mat),
    pts = add_xy.(add_xy.(p, [+leg_x, +leg_x, -leg_x, -leg_x], [-leg_y, +leg_y, +leg_y, -leg_y]), -leg_thickness/2, -leg_thickness/2),
    legs = [b_box(b, pt, leg_thickness, leg_thickness, height - top_thickness, mat) for pt in pts]
  [table_top, legs]
  end

export b_chair
b_chair(b::Backend, p, length, width, height, seat_height, thickness, mat) =
  [b_table(b, p, length, width, seat_height, thickness, thickness, mat),
   b_box(b, add_xyz(p, -length/2, -width/2, seat_height), thickness, width, height - seat_height, mat)]

#=
To avoid an explosion of parameters, we expect functions as arguments, e.g.
b_table_and_chairs(b,
  loc_from_o_phi(p, α),
  p->b_table(b, p, t_length, t_width, t_height, t_top_thickness, t_leg_thickness, mat),
  p->b_chair(b, p, c_length, c_width, c_height, c_seat_height, c_thickness, mat),
  l_length,
  t_width,
  chairs_top,
  chairs_bottom,
  chairs_right,
  chairs_left,
  spacing)

As an example:
b_table_and_chairs(blender,
  loc_from_o_phi(xy(1,2), pi/3),
  p->b_table(blender, p, 3, 1, 1, 0.1, 0.1, -1),
  p->b_chair(blender, p, 0.7, 0.7, 1.5, 0.7, 0.05, -1),
  3, 1, 1, 1, 2, 2, 1)
=#
b_table_and_chairs(b::Backend, p, table::Function, chair::Function,
                   table_length, table_width,
                   chairs_on_top, chairs_on_bottom,
                   chairs_on_right, chairs_on_left,
                   spacing) =
  let dx = table_length/2,
      dy = table_width/2,
      row(p, angle, n) = [loc_from_o_phi(add_pol(p, i*spacing, angle), angle+pi/2) for i in 0:n-1],
      centered_row(p, angle, n) = row(add_pol(p, -spacing*(n-1)/2, angle), angle, n)
    [table(p),
     chair.(centered_row(add_x(p, -dx), -pi/2, chairs_on_bottom))...,
     chair.(centered_row(add_x(p, +dx), +pi/2, chairs_on_top))...,
     chair.(centered_row(add_y(p, +dy), -pi, chairs_on_right))...,
     chair.(centered_row(add_y(p, -dy), 0, chairs_on_left))...]
  end

b_curtain_wall_element(b::Backend, path, bottom, height, l_thickness, r_thickness, family) =
  b_wall_no_openings(b, translate(path, vz(bottom)), height, l_thickness, r_thickness, family)

#@bdef curtain_wall(s, path, bottom, height, thickness, kind)

backend_fill(b, path) =
  backend_fill_curves(b, backend_stroke(b, path))

backend_frame_at(b, c, t) = throw(UndefinedBackendException())
backend_fill_curves(b, ids) = throw(UndefinedBackendException())

#@bdef fill(path::ClosedPathSequence)
backend_fill(b::Backend, path::ClosedPathSequence) =
  backend_fill_curves(b, map(path->backend_stroke(b, path), path.paths))


@bdef ground(level::Loc, color::RGB)

@bdef name()

# Compute bounding box from all shape proxies stored in the backend
function shapes_bbox(b)
  bmin = [Inf, Inf, Inf]
  bmax = [-Inf, -Inf, -Inf]
  for shape in keys(b.refs.shapes)
    for loc in shape_locs(shape)
      let wp = in_world(loc)
        bmin[1] = min(bmin[1], wp.x)
        bmin[2] = min(bmin[2], wp.y)
        bmin[3] = min(bmin[3], wp.z)
        bmax[1] = max(bmax[1], wp.x)
        bmax[2] = max(bmax[2], wp.y)
        bmax[3] = max(bmax[3], wp.z)
      end
    end
  end
  (bmin, bmax)
end

#=
Backends might not provide camera information. In that case
we need to provide it in the frontend.
=#
export ViewType, FrontendView, BackendView, view_type
abstract type ViewType end
struct FrontendView <: ViewType end
struct BackendView <: ViewType end

# By default, we use the backend view
view_type(::Type{<:Backend}) = BackendView()

export b_zoom_extents
b_zoom_extents(b::Backend) = b_zoom_extents(view_type(typeof(b)), b)
b_zoom_extents(::BackendView, b) = missing_specialization(b, :b_zoom_extents)
b_zoom_extents(::FrontendView, b) =
  let (bmin, bmax) = shapes_bbox(b)
    if all(isfinite, bmin) && all(isfinite, bmax)
      let center = xyz((bmin[1]+bmax[1])/2, (bmin[2]+bmax[2])/2, (bmin[3]+bmax[3])/2),
          diag = sqrt((bmax[1]-bmin[1])^2 + (bmax[2]-bmin[2])^2 + (bmax[3]-bmin[3])^2),
          dist = max(diag * 1.5, 10.0),
          camera = center + vxyz(dist*0.6, dist*0.6, dist*0.5)
        b.view.camera = camera
        b.view.target = center
      end
    end
  end

export b_set_ground
b_set_ground(b::Backend, level, mat) =
  b_surface_regular_polygon(b, 16, z(level), 10000, 0, true, material_ref(b, mat))

b_realistic_sky(b::Backend, date, latitude, longitude, elevation, meridian, turbidity, sun) =
  b_realistic_sky(b, sun_pos(date, meridian, latitude, longitude)..., turbidity, sun)

# Rendering

export b_render_pathname, b_render_initial_setup, b_render_final_setup, b_setup_render, b_render_view

b_setup_render(b::Backend, kind) = kind

b_render_view(b::Backend, name) =
  let path = prepare_for_saving_file(b_render_pathname(b, name))
    b_render_final_setup(b, render_kind())
    b_render_and_save_view(b, path)
  end

prepare_for_saving_file(path::String) =
  let p = normpath(path)
    mkpath(dirname(p))
    try
      rm(p, force=true)
      p
    catch e
      if isa(e, Base.IOError)
        let (base, ext) = splitext(path)
          if ext == ".pdf"
            prepare_for_saving_file(base*"_"*ext)
          else
            throw(e)
          end
        end
      else
        throw(e)
      end
    end
  end

b_render_pathname(::Backend, name) = render_default_pathname(name)  
b_render_initial_setup(::Backend, kind) = kind
b_render_final_setup(::Backend, kind) = kind

@bdef b_render_and_save_view(path)

# -- shot_view: fast viewport capture to raster image (PNG) --
export b_shot_view, b_shot_pathname, b_raw_view, b_raw_pathname

b_shot_view(b::Backend, path) = b_render_and_save_view(b, path)
b_shot_pathname(b::Backend, name) = b_render_pathname(b, name)

# -- raw_view: capture native intermediate format for precise comparison --
b_raw_view(b::Backend, path) = b_shot_view(b, path)
b_raw_pathname(b::Backend, name) = b_shot_pathname(b, name)

# Viewport: Only some backends support this, but it can be useful for the frontend to be able to set it in a backend-agnostic way.
b_set_view_size(b::Backend, width, height) = nothing

#######

b_right_cuboid(b::Backend, cb, width, height, h, mat) =
  b_box(b, add_xy(cb, -width/2, -height/2), width, height, h, mat)


backend_stroke(b::Backend, path::Union{OpenPathSequence,ClosedPathSequence}) =
  backend_stroke_unite(b, map(path->backend_stroke(b, path), path.paths))
backend_stroke(b::Backend, path::PathOps) =
  begin
      start, curr, refs = path.start, path.start, new_refs(b)
      for op in path.ops
          start, curr, refs = backend_stroke_op(b, op, start, curr, refs)
      end
      if path.closed
          push!(refs, backend_stroke_line(b, [curr, start]))
      end
      backend_stroke_unite(b, refs)
  end

backend_stroke(b::Backend, path::PathSet) =
    for p in path.paths
        backend_stroke(b, p)
    end

@bdef stroke_op(op::LineXThenYOp, start::Loc, curr::Loc, refs)
@bdef stroke_op(op::LineYThenXOp, start::Loc, curr::Loc, refs)
#@bdef stroke_op(op::MoveOp, start::Loc, curr::Loc, refs)
#@bdef stroke_op(op::MoveToOp, start::Loc, curr::Loc, refs)

#@bdef stroke_op(op::LineOp, start::Loc, curr::Loc, refs)
backend_stroke_op(b::Backend, op::LineOp, start::Loc, curr::Loc, refs) =
  (start, curr + op.vec, push!(refs, backend_stroke_line(b, [curr, curr + op.vec])))

#backend_stroke_op(b::Backend, op::CloseOp, start::Loc, curr::Loc, refs) =
#    (start, start, push!(refs, backend_stroke_line(b, [curr, start])))

#@bdef stroke_op(op::ArcOp, start::Loc, curr::Loc, refs)
backend_stroke_op(b::Backend, op::ArcOp, start::Loc, curr::Loc, refs) =
  let center = curr - vpol(op.radius, op.start_angle)
    (start, center + vpol(op.radius, op.start_angle + op.amplitude),
     push!(refs, backend_stroke_arc(b, center, op.radius, op.start_angle, op.amplitude)))
  end


#

@bdef stroke_unite(refs)
#@bdef surface_boundary(s::Shape2D)
#@bdef surface_domain(s::Shape2D)

#@bdef wall(path, height, l_thickness, r_thickness, family)

# A poor's man approach to deal with Z-fighting
export support_z_fighting_factor, wall_z_fighting_factor
const support_z_fighting_factor = 0.999
const wall_z_fighting_factor = 0.998

@bdef wall_path(path::OpenPolygonalPath, height, l_thickness, r_thickness)
@bdef wall_path(path::Path, height, l_thickness, r_thickness)

# Select things from a backend
@bdef(b_select_position(prompt))
@bdef(b_select_positions(prompt))
@bdef(b_select_point(prompt))
@bdef(b_select_points(prompt))
@bdef(b_select_curve(prompt))
@bdef(b_select_curves(prompt))
@bdef(b_select_surface(prompt))
@bdef(b_select_surfaces(prompt))
@bdef(b_select_solid(prompt))
@bdef(b_select_solids(prompt))
@bdef(b_select_shape(prompt))
@bdef(b_select_shapes(prompt))

# Illustrations

@bdef(b_radius_illustration(c, r, c_text, r_text))
@bdef(b_arc_illustration(c, r, s, a, r_text, s_text, a_text))

#=
The backend might be in the same process as the frontend, or
it might be in a different process.
=#
export TargetType, LocalTarget, RemoteTarget
abstract type TargetType end
struct LocalTarget <: TargetType end
struct RemoteTarget <: TargetType end

# By default, we use a local target
target_type(::Type{<:Backend}) = LocalTarget()
target(b::T) where T = target(target_type(T), b)

target(::LocalTarget, b) = b.target
target(::RemoteTarget, b) = b.connection()

b_set_view(b::T, camera, target, lens, aperture) where {T<:Backend} =
  b_set_view(view_type(T), b, camera, target, lens, aperture)

b_set_view_top(b::T) where {T<:Backend} = b_set_view_top(view_type(T), b)

b_get_view(b::T) where {T<:Backend} = b_get_view(view_type(T), b)

###############################################################
# For the frontend, we assume there is a property to store the 
# view details. A simple approach is to use a mutable struct

mutable struct View
  camera::Loc
  target::Loc
  lens::Real
  aperture::Real
  is_top_view::Bool
end

default_view() = View(xyz(10,10,10), xyz(0,0,0), 35, 22, false)
top_view() = View(xyz(0,0,10), xyz(0,0,0), 0, 0, true)

export View, default_view, top_view, b_get_view, b_set_view, b_set_view_top


b_set_view(::FrontendView, b, camera, target, lens, aperture) =
  begin
    b.view.camera = camera
    b.view.target = target
    b.view.lens = lens
    b.view.aperture = aperture
    b.view.is_top_view = norm(cross(target - camera, vz(1, world_cs))) < 1e-9  # aligned with Z?
  end

b_set_view_top(::FrontendView, b) =
  begin
    b.view.camera = z(1000)
    b.view.target = z(0)
    b.view.lens = 1000
    b.view.is_top_view = true
  end

# For legacy reasons, we only return camera, target, and lens.
b_get_view(::FrontendView, b) =
  b.view.camera, b.view.target, b.view.lens

###############################################################
# For the backend, we always rely on the b_set_view function.

b_set_view_top(v::BackendView, b) =
  b_set_view(b, xyz(0.0,0.0,10), xyz(0.0,0.1,0.0), 50, 0)

###############################################################
# It might be useful to stop view update during batch processing
export with_batch_processing, b_start_batch_processing, b_stop_batch_processing

b_start_batch_processing(b::Backend) = nothing

b_stop_batch_processing(b::Backend) = nothing

with_batch_processing(f) =
  try
    foreach(b_start_batch_processing, current_backends())
    f()
  finally
    foreach(b_stop_batch_processing, current_backends())
  end


