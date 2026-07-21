# CodeGen.jl — Portable BIM-model → Julia-program code generation.
#
# This is the backend-agnostic half of the "introspect a CAD/BIM model and emit an equivalent
# Khepri program" capability. It operates purely on portable Khepri BIM shapes (Wall, Slab, Column,
# …) and `Expr`, and was lifted verbatim out of KhepriRevit (Revit.jl) so that any backend can reuse
# it. The only backend-specific decisions are isolated behind two hooks:
#
#   b_native_family_expr(b, var, meta)  — the `set_backend_family(...)` line(s) a backend emits for a
#                                         reconstructed family (nothing ⇒ no mapping).
#   b_codegen_module(b)                 — the module named in the emitted `using` line.
#
# Architecture (inspired by Tomás Grelha da Cunha's MSc thesis, "Refactoring for Dynamic Languages:
# The Julia Case", IST 2020):
#   1. Introspect : backend model → Khepri shape objects   (backend-specific; see Introspection.jl)
#   2. meta_program: shapes → raw Julia Expr (AST)
#   3. Transform  : refactoring passes Expr → Expr (extract variables, loop rerolling, …)
#   4. Print      : pretty-print final Expr → source text

# ─── Backend hooks (defaults) ─────────────────────────────────────────────────

# The native `set_backend_family(...)` expression for a reconstructed family; `nothing` ⇒ emit none.
# `meta` is the introspection metadata NamedTuple stored for the family. Backends with a native family
# system (Revit) override this; geometry-only backends leave the default.
b_native_family_expr(b::Backend, var, meta) = nothing

# The module named in the emitted `using` line. Default derives it from the backend's package; socket
# backends (whose concrete type lives in KhepriBase) must override.
b_codegen_module(b::Backend) = nameof(parentmodule(typeof(b)))

# ─── Utilities ────────────────────────────────────────────────────────────────

sanitize_name(name) =
  let s = replace(lowercase(strip(name)), r"[^a-z0-9_]" => "_"),
      s = replace(s, r"_+" => "_"),
      s = strip(s, '_')
    isempty(s) ? "unnamed" : (isdigit(s[1]) ? "_" * s : s)
  end

# Bottom-up map over an Expr tree
map_expr(f, e::Expr) = f(Expr(e.head, map(x -> map_expr(f, x), e.args)...))
map_expr(f, x) = f(x)

# Collect all sub-expressions matching a predicate
collect_exprs(pred, x) = pred(x) ? [x] : []
collect_exprs(pred, e::Expr) =
  let result = pred(e) ? [e] : []
    for a in e.args
      append!(result, collect_exprs(pred, a))
    end
    result
  end

# Compare two Exprs structurally, returning list of (path, val1, val2) diffs.
# path is a vector of arg indices.
expr_diff(e1, e2) = e1 == e2 ? [] : [([],  e1, e2)]
expr_diff(e1::Expr, e2::Expr) =
  if e1.head != e2.head || length(e1.args) != length(e2.args)
    [([], e1, e2)]
  else
    let diffs = []
      for (i, (a1, a2)) in enumerate(zip(e1.args, e2.args))
        for (path, v1, v2) in expr_diff(a1, a2)
          push!(diffs, (vcat([i], path), v1, v2))
        end
      end
      diffs
    end
  end

# Replace the value at a path in an Expr with a new value
expr_replace_at(e, path, val) =
  isempty(path) ? val :
  let args = collect(e.args)
    args[path[1]] = expr_replace_at(args[path[1]], path[2:end], val)
    Expr(e.head, args...)
  end

# The family constructor names known to the codegen passes (dedup/naming/section grouping).
const _family_function_names = Set([:wall_family, :curtain_wall_family, :slab_family, :ceiling_family,
                    :column_family, :beam_family, :door_family, :window_family,
                    :roof_family, :family_element_family, :table_family, :chair_family,
                    :toilet_family, :sink_family, :closet_family,
                    :stair_family, :railing_family])

#=
LOCAL-coordinate contexts: subtrees whose xy/xyz leaves are NOT world positions
— wall-relative opening positions (add_door/add_window and the doors=/windows=
kwargs) and family-local frame profiles (family constructors and the family=
kwarg). The translate/anchor/collect passes must never rebase inside them.
One predicate, consumed by translate_xyz_expr, _anchor_relative_expr and
_collect_world_xy, so the three can never drift apart again.
=#
_local_coord_context(e::Expr) =
  (e.head == :kw && e.args[1] in (:doors, :windows, :family)) ||
  (e.head == :call && (e.args[1] in (:add_door, :add_window) || e.args[1] in _family_function_names))

#=
Constructs whose interior numeric leaves must never be parametrized or aliased
(unify_constants, extract_shared_dimensions): families and their frame_family
casing profiles, levels (incl. unconnected markers), opening positions, backend
mappings, groups and their placements, obj fallbacks, resource registration.
frame_family is here for a subtle reason: hoist_opening_frames lifts it OUT of
the door/window family constructor whose protection used to cover it
transitively — hoisted, it must be protected by name, or a wall dimension that
happens to equal the casing width gets aliased into the frame profile.
NOTE: _clone_eligible's list is intentionally DIFFERENT (it decides what may be
lifted into functions, not what must stay literal) — do not merge them.
=#
const _protected_callee_set =
  Set([_family_function_names...,
       :frame_family, :level, :unconnected_level, :add_door, :add_window,
       :set_backend_family, :obj_model, :group, :group_instance,
       :add_resource_folder!])
_protected_callee(c) = c in _protected_callee_set

# Family metadata is carried explicitly in `model.family_meta` (an IdDict family → FamilyMeta) built
# by introspection, not in module globals — see family_expr_map below.

# `is_curtain_wall` is the @defproxy-generated predicate for CurtainWall (Shapes.jl); reused as-is.

# Deduplicate slabs whose boundary vertices are the same set (different order)
_dedup_slabs(members) =
  let seen_slab_verts = Set{Set{Tuple{Float64,Float64,Float64}}}(),
      result = []
    for m in members
      if m isa Slab
        let verts = m.region isa Region ?
              (m.region.paths[1] isa ClosedPolygonalPath ?
                Set((round(cx(v), digits=4), round(cy(v), digits=4), round(cz(v), digits=4))
                    for v in m.region.paths[1].vertices) :
                Set{Tuple{Float64,Float64,Float64}}()) :
              Set{Tuple{Float64,Float64,Float64}}()
          if isempty(verts) || verts ∉ seen_slab_verts
            push!(seen_slab_verts, verts)
            push!(result, m)
          end
        end
      else
        push!(result, m)
      end
    end
    result
  end

# NB: CurtainWall is its own proxy type, NOT a Wall subtype — test the predicate first, or curtain
# walls classify as :other and their family variables get an "other_" prefix.
_shape_family_category(m) =
  is_curtain_wall(m) ? :curtain_wall :
  m isa Wall ? :wall :
  m isa Slab ? :slab :
  m isa Column ? :column :
  m isa FreeColumn ? :column :
  m isa Beam ? :beam :
  m isa Ceiling ? :ceiling :
  m isa Roof ? :roof :
  m isa Stair ? :stair :
  m isa Railing ? :railing :
  m isa Table ? :table :
  m isa Chair ? :chair :
  m isa Toilet ? :toilet :
  m isa Sink ? :sink :
  m isa Closet ? :closet :
  m isa FamilyElement ? :family_element : :other

# ─── Phase 2: Model → Expr ────────────────────────────────────────────────────

# Map each reconstructed family's meta_program Expr to its metadata, from the model's `family_meta`
# carrier (IdDict family → FamilyMeta). Because families are reconstructed *named by their type*
# (family_key), distinct native types produce distinct Exprs here, so this map does NOT collapse
# them the way the former Expr-keyed global did. Consumed by extract_families (variable naming) and
# add_backend_families (native family mapping).
family_expr_map(model) =
  let fmap = Dict{Expr, FamilyMeta}(),
      fm = hasproperty(model, :family_meta) ? model.family_meta : ()
    for (fam, meta) in fm
      fmap[meta_program(fam)] = meta
    end
    fmap
  end

function model_to_expr(model)
  stmts = Expr[]
  for l in model.levels
    push!(stmts, meta_program(l))
  end
  for w in model.walls
    push!(stmts, meta_program(w))
  end
  for f in model.floors
    push!(stmts, meta_program(f))
  end
  for c in model.columns
    push!(stmts, meta_program(c))
  end
  for bm in model.beams
    push!(stmts, meta_program(bm))
  end
  for ce in model.ceilings
    push!(stmts, meta_program(ce))
  end
  for rf in model.roofs
    push!(stmts, meta_program(rf))
  end
  for fx in model.fixtures
    push!(stmts, meta_program(fx))
  end
  for st in model.stairs
    push!(stmts, meta_program(st))
  end
  for rl in model.railings
    push!(stmts, meta_program(rl))
  end
  # Groups: emit group definitions and group_instance placements
  for g in model.groups
    let grp_name = sanitize_name("group_$(g.type_name)"),
        grp_var = Symbol(grp_name),
        factory_var = Symbol("$(grp_name)_factory"),
        # Compute group origin from first instance location
        origin = g.instances[1],
        # Build member shape expressions, translated to be group-relative. Z is NOT translated:
        # member heights live entirely in their levels/level-relative coordinates, so instances
        # carry an xy offset only — a z here would double-count against the levels on mesh
        # backends (Revit ignores curve z, which masked the imbalance).
        member_stmts = Expr[]
      for m in g.members
        push!(member_stmts, meta_program(m))
      end
      body_expr = Expr(:block, member_stmts...)
      translated_body = translate_xyz_expr(body_expr, cx(origin), cy(origin), 0.0)
      # Emit: factory function
      push!(stmts, Expr(:function, Expr(:call, factory_var), translated_body))
      # Emit: grp_var = group("name", factory=factory_var)
      push!(stmts, Expr(:(=), grp_var,
        Expr(:call, :group,
          Expr(:parameters,
            Expr(:kw, :factory, factory_var)),
          grp_name)))
      # Emit: group_instance(grp_var, xyz(x, y, 0)) — xy placement only; see the z note above.
      for loc in g.instances
        push!(stmts, Expr(:call, :group_instance, grp_var,
          Expr(:call, :xyz,
            meta_program(cx(loc)), meta_program(cy(loc)), 0.0)))
      end
    end
  end
  # Mesh-fallback tail (Phase 6): emit an obj_model placement for each element no parametric reader
  # consumed (curtain panels, MEP, topography, in-place, sloped/degenerate), so nothing is silently lost.
  if hasproperty(model, :fallback_meshes)
    for m in model.fallback_meshes
      push!(stmts, meta_program(m))
    end
  end
  # Deferred-group flush as the LAST statement: backends that postpone group construction (Revit)
  # build them here, when no further element creation can disturb them. No-op elsewhere.
  isempty(model.groups) || push!(stmts, Expr(:call, :finalize_groups))
  Expr(:block, stmts...)
end

# ── Geometric round-trip (non-BIM backends: AutoCAD, …) ──────────────────────
# Emit a geometry-only program: one meta_program statement per reconstructed shape. Skips shapes that
# can't round-trip to reproducible code — `Unknown` (opaque native handles, e.g. an AutoCAD Solid3d the
# CAD API cannot classify back to box/sphere/…), which would otherwise emit a dead `unknown(handle)`
# call. Needs no BIM structure; the standard passes then refine it (extract_levels/families no-op with
# none present, while loop_rerolling still rerolls repeated geometry such as arrays into for-loops).
shapes_to_expr(shapes) =
  let keep = [s for s in shapes if !is_unknown(s)],
      dropped = length(shapes) - length(keep)
    dropped > 0 && @info "shapes_to_expr: skipped $dropped non-reproducible shape(s) (e.g. AutoCAD Solid3d → unknown handle)"
    Expr(:block, [meta_program(s) for s in keep]...)
  end

# Run the full codegen pipeline on an already-reconstructed set of geometric shapes (e.g. from
# existing_shapes on a geometric backend). Same passes as the BIM generate_khepri_code; the family passes
# no-op without families. Returns the source; also writes it to output_path when a path is given.
function shapes_to_khepri_code(shapes, b; output_path=nothing, wrap=false)
  let raw_expr = shapes_to_expr(shapes),
      fmap = Dict{Expr, FamilyMeta}(),
      passes = codegen_passes(b, fmap; header=add_geometric_header(b), wrap=wrap),
      code = expr_to_string(foldl((e, pass) -> pass(e), passes, init=raw_expr))
    output_path === nothing || open(io -> write(io, code), output_path, "w")
    code
  end
end

# Translate all xyz/xy calls in an Expr tree by subtracting an offset.
# Skips door/window positions (wall-relative, not world coords).
translate_xyz_expr(x, dx, dy, dz) = x
translate_xyz_expr(e::Expr, dx, dy, dz) =
  if _local_coord_context(e)
    # These carry LOCAL coordinates — wall-relative opening positions and family-local frame
    # profiles — not world positions, so they must not be shifted to group-relative coordinates.
    e
  elseif e.head == :call && e.args[1] == :xyz && length(e.args) == 4 &&
     all(a -> a isa Real, e.args[2:4])
    Expr(:call, :xyz,
      meta_program(e.args[2] - dx),
      meta_program(e.args[3] - dy),
      meta_program(e.args[4] - dz))
  elseif e.head == :call && e.args[1] == :xy && length(e.args) == 3 &&
         all(a -> a isa Real, e.args[2:3])
    Expr(:call, :xy,
      meta_program(e.args[2] - dx),
      meta_program(e.args[3] - dy))
  else
    Expr(e.head, [translate_xyz_expr(a, dx, dy, dz) for a in e.args]...)
  end

#=
Anchor-relative rewrite — TOTAL: every WORLD position construction becomes an
anchor-relative one, whatever its coordinate arguments are.
  xy(A, B)     → add_xy(anchor, A⊖px, B⊖py)
  xyz(A, B, C) → add_xyz(anchor, A⊖px, B⊖py, C)
where ⊖ folds to a rounded literal when the coordinate is a literal and stays a
runtime subtraction (`A - px`) when it bears a loop variable, a parameter symbol
introduced by unify_constants, or a cluster parameter — so rerolled series and
symbol-bearing coordinates re-site exactly like literal ones. add_xy/add_xyz
offset in the ANCHOR's coordinate system: with the default (world) anchor the
geometry is unchanged, and an oriented anchor rotates every placement. z is a
level-relative height: add_xyz adds it to the anchor's z, so a z-bearing anchor
uniformly lifts loose (Loc-placed) geometry while level-bound heights stay with
the levels — the anchor is a PLAN anchor. Element self-rotation arguments
(column/family_element angles) do NOT follow an oriented anchor; documented
limitation until the rotation-compensation follow-up.
The _local_coord_context skip (opening positions + family-local frames) is
shared with translate_xyz_expr/_collect_world_xy; group factory bodies are
skipped by the caller (already instance-relative — rebasing would double-shift
against the rebased group_instance placement).
=#
_anchor_delta(a, off) =
  a isa Real ? meta_program(a - off) :
  off == 0 ? a :
  off < 0 ? Expr(:call, :+, a, meta_program(-off)) : Expr(:call, :-, a, off)

_anchor_relative_expr(x, px, py, anchor) = x
_anchor_relative_expr(e::Expr, px, py, anchor) =
  if _local_coord_context(e)
    e
  elseif e.head == :call && e.args[1] == :xyz && length(e.args) == 4
    Expr(:call, :add_xyz, anchor,
         _anchor_delta(e.args[2], px), _anchor_delta(e.args[3], py), e.args[4])
  elseif e.head == :call && e.args[1] == :xy && length(e.args) == 3
    Expr(:call, :add_xy, anchor,
         _anchor_delta(e.args[2], px), _anchor_delta(e.args[3], py))
  else
    Expr(e.head, [_anchor_relative_expr(a, px, py, anchor) for a in e.args]...)
  end

# Defensive invariant for the total rewrite: after rebasing, NO world position
# construction may survive outside the local-coordinate contexts. New
# meta_program coordinate forms must be routed through the rewriter (and the
# oracle fixtures) — failing loud here beats a storey that silently tears when
# re-sited.
function _assert_no_world_xy(e, ctx)
  bad = Ref{Any}(nothing)
  walk(x) = nothing
  walk(ex::Expr) =
    _local_coord_context(ex) ? nothing :
    ex.head == :call && (ex.args[1] == :xy || ex.args[1] == :xyz) ?
      (bad[] = ex; nothing) :
      (foreach(walk, ex.args); nothing)
  walk(e)
  bad[] === nothing ||
    error("sectionalize_by_storey: absolute world coordinate survived the anchor rebase in $(ctx): $(bad[])")
end

# Collect (x, y) of every LITERAL world position (same skip list) — used to pick the building
# origin corner. Symbol-bearing coordinates (loop variables, parameter symbols) cannot contribute
# to a static minimum; they still rebase (the rewrite is total), so the origin is the SW corner of
# the literal-coordinate geometry, not necessarily of everything.
_collect_world_xy(x, acc) = nothing
function _collect_world_xy(e::Expr, acc)
  _local_coord_context(e) && return nothing
  if e.head == :call && e.args[1] in (:xy, :xyz) && length(e.args) >= 3 &&
     e.args[2] isa Real && e.args[3] isa Real
    push!(acc, (Float64(e.args[2]), Float64(e.args[3])))
  end
  for a in e.args
    _collect_world_xy(a, acc)
  end
  nothing
end

# ─── Phase 3: Transform Passes ────────────────────────────────────────────────

# 3.1 Extract levels: deduplicate level(h) calls into named variables
function extract_levels(e::Expr)
  level_calls = collect_exprs(
    x -> x isa Expr && x.head == :call && x.args[1] in (:level, :unconnected_level) &&
         length(x.args) == 2,
    e)
  unique_levels = unique(level_calls)
  isempty(unique_levels) && return e
  # Sort by height value; non-literal heights (e.g. already-parametrized expressions) sort after
  # literals by their printed form, so numbering stays deterministic instead of tying at 0.0.
  sorted = sort(unique_levels,
                by=x -> x.args[2] isa Real ? (0, Float64(x.args[2]), "") : (1, 0.0, string(x.args[2])))
  assignments = Expr[]
  replacements = Dict{Expr, Symbol}()
  for (i, lc) in enumerate(sorted)
    var = Symbol("level_$(i-1)")
    push!(assignments, Expr(:(=), var, lc))
    replacements[lc] = var
  end
  body = map_expr(x -> get(replacements, x, x), e)
  # Filter out bare symbols left over from replacing standalone level() calls
  level_syms = Set(values(replacements))
  body_stmts = filter(s -> !(s isa Symbol && s in level_syms), body.args)
  Expr(:block, assignments..., body_stmts...)
end

# 3.2 Extract families: deduplicate family constructors into named variables
extract_families(fmap) = function (e::Expr)
  family_fns = _family_function_names
  family_calls = collect_exprs(
    x -> x isa Expr && x.head == :call && x.args[1] in family_fns,
    e)
  unique_families = unique(family_calls)
  isempty(unique_families) && return e
  assignments = Expr[]
  replacements = Dict{Expr, Symbol}()
  used_names = Dict{Symbol, Int}()
  for fc in unique_families
    # Use introspection metadata for descriptive variable names
    meta = get(fmap, fc, nothing)
    var = if meta !== nothing
      Symbol(sanitize_name("$(meta.category)_$(meta.family_name)_$(meta.type_name)"))
    else
      let cat = replace(string(fc.args[1]), "_family" => "")
        Symbol("$(cat)_fam_$(length(assignments)+1)")
      end
    end
    # Two distinct types whose names differ only in case/punctuation sanitize to the same variable;
    # without a counter the second assignment silently shadows the first and every element bound to
    # family A would reconstruct with family B.
    let n = get(used_names, var, 0)
      used_names[var] = n + 1
      n > 0 && (var = Symbol(var, :_, n + 1))
    end
    push!(assignments, Expr(:(=), var, fc))
    replacements[fc] = var
  end
  # map_expr replaces bottom-up, so a family whose argument is itself a family call
  # (stair_family(..., railing_family(...))) has its inner call replaced by a variable
  # BEFORE the outer expr is visited — the mutated outer expr no longer matches its
  # key and stays inline (unextracted, unguarded). Register the inner-replaced form of
  # each key as an alias for the same variable; repeat to a fixpoint for deeper nests.
  let changed = true
    while changed
      changed = false
      for (fc, var) in collect(replacements)
        let normalized = Expr(fc.head, fc.args[1],
                              [map_expr(x -> get(replacements, x, x), a)
                               for a in fc.args[2:end]]...)
          if normalized != fc && !haskey(replacements, normalized)
            replacements[normalized] = var
            changed = true
          end
        end
      end
    end
  end
  stmts = e.head == :block ? e.args : [e]
  # Find where levels end and elements start
  first_non_assign = findfirst(s -> !_is_any_level_assign(s), stmts)
  insert_pos = first_non_assign === nothing ? length(stmts) + 1 : first_non_assign
  body = map_expr(x -> get(replacements, x, x), Expr(:block, stmts...))
  Expr(:block, body.args[1:insert_pos-1]..., assignments..., body.args[insert_pos:end]...)
end

# 3.2b Hoist the shared opening frame — every door/window family embeds an identical frame_family(...)
# sub-expression (the casing profile of the opening). Lift each frame_family value recurring across
# ≥2 families into a named `opening_frame` variable and share it, so the profile is defined once.
# frame_family is not in _family_function_names, so extract_families leaves it inline; this runs right
# after it. Geometry-neutral: only replaces identical sub-expressions with an equal-valued binding.
# The standard symmetric casing profile rectangular_path(xy(-w/2, -w/2), w, w) additionally gets its
# width lifted to a frame_width parameter (the Isenberg frame_width idiom — widening the casing is
# one edit, not three coordinated ones inside a single expression).
_frame_width_of(f::Expr) =
  let prof = length(f.args) >= 3 ? f.args[3] : nothing
    prof isa Expr && prof.head == :call && prof.args[1] == :rectangular_path &&
    length(prof.args) == 4 && prof.args[3] isa AbstractFloat && prof.args[3] == prof.args[4] &&
    prof.args[2] isa Expr && prof.args[2].head == :call && prof.args[2].args[1] == :xy &&
    all(a -> a isa Real && abs(a + prof.args[3] / 2) < 1e-9, prof.args[2].args[2:3]) ?
      Float64(prof.args[3]) : nothing
  end

_frame_with_width(f::Expr, wsym) =
  let half = Expr(:call, :-, Expr(:call, :/, wsym, 2))
    Expr(:call, f.args[1], f.args[2],
         Expr(:call, :rectangular_path, Expr(:call, :xy, half, half), wsym, wsym),
         f.args[4:end]...)
  end

function hoist_opening_frames(e::Expr)
  frames = collect_exprs(x -> x isa Expr && x.head == :call && x.args[1] == :frame_family, e)
  isempty(frames) && return e
  counts = Dict{Expr, Int}()
  for f in frames
    counts[f] = get(counts, f, 0) + 1
  end
  repeated = sort([f for (f, c) in counts if c >= 2], by = string)
  isempty(repeated) && return e
  used = Set{Symbol}(s.args[1] for s in stmts_of(e)
                     if s isa Expr && s.head == :(=) && s.args[1] isa Symbol)
  table = Dict{Expr, Symbol}()
  assigns = Expr[]
  for (i, f) in enumerate(repeated)
    let name = _fresh_param(i == 1 ? "opening_frame" : "opening_frame_$(i)", used),
        w = _frame_width_of(f)
      if w === nothing
        push!(assigns, Expr(:(=), name, f))
      else
        let wsym = _fresh_param(i == 1 ? "frame_width" : "frame_width_$(i)", used)
          push!(assigns, Expr(:(=), wsym, w))
          push!(assigns, Expr(:(=), name, _frame_with_width(f, wsym)))
        end
      end
      table[f] = name
    end
  end
  stmts = collect(stmts_of(map_expr(x -> get(table, x, x), e)))
  famassign = s -> s isa Expr && s.head == :(=) && s.args[2] isa Expr &&
                   s.args[2].head == :call && s.args[2].args[1] in _family_function_names
  pos = something(findfirst(famassign, stmts), 1)
  Expr(:block, stmts[1:pos-1]..., assigns..., stmts[pos:end]...)
end

# Build `@isdefined(backend) && set_backend_family(var, backend, native)`: a cross-backend family
# mapping that applies only when its backend is actually loaded. This lets one generated program carry
# mappings for several backends at once (a Revit `.rfa` and an extracted OBJ for KhepriThreejs) yet run
# on each without referencing — or evaluating the native constructor of — the others. `&&` short-circuits
# so e.g. `revit_file_family(...)` is never evaluated in a KhepriThreejs session (where it's undefined).
guarded_backend_family_expr(var, backend::Symbol, native) =
  Expr(:&&,
       Expr(:macrocall, Symbol("@isdefined"), nothing, backend),
       Expr(:call, :set_backend_family, var, backend, native))

# 3.3 Add backend family mappings
# Uses _family_expr_meta populated by model_to_expr; the actual native family expression is
# supplied by the backend via b_native_family_expr(b, var, meta).
add_backend_families(b, fmap) = (e::Expr) ->
  let stmts = e.head == :block ? collect(e.args) : [e],
      family_fns = _family_function_names,
      new_stmts = Any[]
    for s in stmts
      push!(new_stmts, s)
      if s isa Expr && s.head == :(=) &&
         s.args[2] isa Expr && s.args[2].head == :call &&
         s.args[2].args[1] in family_fns
        let var = s.args[1],
            rhs = s.args[2],
            meta = get(fmap, rhs, nothing)
          if meta !== nothing
            let backend_call = b_native_family_expr(b, var, meta)
              # A backend may map a family to several backends at once (e.g. Revit `.rfa` + extracted
              # OBJ for KhepriThreejs), so accept a list of statements as well as a single one.
              if backend_call isa AbstractVector
                append!(new_stmts, backend_call)
              elseif backend_call !== nothing
                push!(new_stmts, backend_call)
              end
            end
          end
        end
      end
    end
    Expr(:block, new_stmts...)
  end

# 3.4 Loop rerolling: detect repeated shape calls forming grid patterns
loop_rerolling(e::Expr) =
  let stmts = e.head == :block ? collect(e.args) : [e],
      result = Any[],
      # Group consecutive statements by their "template" (shape call with same structure)
      i = 1
    while i <= length(stmts)
      let s = stmts[i]
        if s isa Expr && (s.head == :call || s.head == :let)
          # A run = consecutive statements with the same tree shape and equal non-numeric leaves
          # (same callee, same family variable, same arity — only numbers vary). Statements whose
          # symbols differ (e.g. a different family) end the run, so each family's subsequence can
          # reroll on its own. `:let` walls (wall + add_door/add_window) participate like calls.
          let run_end = i,
              template = s
            for j in (i+1):length(stmts)
              if stmts[j] isa Expr && _same_shape(template, stmts[j])
                run_end = j
              else
                break
              end
            end
            if run_end - i + 1 >= 4  # Need at least 4 similar statements
              let run = stmts[i:run_end],
                  rerolled = _try_reroll(run)
                if rerolled !== nothing
                  push!(result, rerolled)
                  i = run_end + 1
                  continue
                end
              end
            end
          end
        end
        push!(result, s)
        i += 1
      end
    end
    Expr(:block, result...)
  end

function _try_reroll(stmts)
  length(stmts) < 4 && return nothing
  template = stmts[1]
  all_diffs = [expr_diff(template, s) for s in stmts[2:end]]
  isempty(all_diffs) && return nothing
  # ref_paths = the UNION of every path that varies in ANY consecutive pair. A 2-D grid varies
  # in one axis for some pairs and in both for others, so no single pair sees all the varying
  # paths — the old per-pair-equality test never reached the 2-D detector and left grids flat.
  ref_paths = Set{Vector{Int}}()
  for ds in all_diffs, (p, _, _) in ds
    push!(ref_paths, p)
  end
  paths = sort(collect(ref_paths))
  isempty(paths) && return nothing
  # Read one value per statement at each varying path — from EVERY statement, not just the ones
  # where it happens to differ from the template — so the per-path lists stay aligned and each has
  # length == length(stmts). (`loop_rerolling` only forms a run of structurally-identical calls
  # with leaf-only diffs, so every path is a valid leaf path in every statement.)
  value_lists = Dict{Vector{Int}, Vector{Any}}(
    p => [_get_diff_value(s, p) for s in stmts] for p in paths)
  # Try to detect 2D grid pattern (two varying paths); otherwise a lockstep translation
  # (a run of translated statements varies EVERY coordinate by the same per-statement delta —
  # e.g. repeated walls, whose two endpoints always move together — needing one loop variable).
  if length(paths) == 1
    _try_reroll_1d(template, paths[1], value_lists[paths[1]])
  elseif length(paths) == 2
    let grid = _try_reroll_2d(template, paths, value_lists)
      grid !== nothing ? grid : _try_reroll_lockstep(template, paths, value_lists)
    end
  elseif length(paths) <= 8
    _try_reroll_lockstep(template, paths, value_lists)
  else
    nothing
  end
end

# Lockstep reroll: the first varying path's values form an arithmetic run IN STATEMENT ORDER
# (sort_statements canonicalizes the order beforehand) and every other path is that run plus a
# constant offset. Rerolls diagonal runs and multi-coordinate translations (walls) that the
# grid detector cannot.
function _try_reroll_lockstep(template, paths, value_lists)
  all(vs -> all(_num, vs), values(value_lists)) || return nothing
  let ref = value_lists[paths[1]],
      n = length(ref),
      d = ref[2] - ref[1]
    abs(d) > 1e-9 || return nothing
    all(i -> abs((ref[i] - ref[i-1]) - d) < 0.001, 2:n) || return nothing
    let offsets = Vector{Float64}()
      for p in paths[2:end]
        let vs = value_lists[p],
            c = vs[1] - ref[1]
          all(i -> abs((vs[i] - ref[i]) - c) < 0.001, 1:n) || return nothing
          push!(offsets, c)
        end
      end
      let var = gensym_short(:x),
          body = expr_replace_at(template, paths[1], var)
        for (p, c) in zip(paths[2:end], offsets)
          body = expr_replace_at(body, p,
                                 abs(c) < 1e-9 ? var : Expr(:call, :+, var, meta_program(c)))
        end
        Expr(:for,
             Expr(:(=), var,
                  Expr(:call, :range, meta_program(ref[1]),
                       Expr(:kw, :step, meta_program(d)), Expr(:kw, :length, n))),
             Expr(:block, body))
      end
    end
  end
end

_get_diff_value(e::Expr, path) =
  isempty(path) ? e : _get_diff_value(e.args[path[1]], path[2:end])
_get_diff_value(x, path) = isempty(path) ? x : error("path too deep")

function _try_reroll_1d(template, path, values)
  range = _try_make_range(values)
  range === nothing && return nothing
  let var = gensym_short(:i),
      body = expr_replace_at(template, path, var)
    Expr(:for, Expr(:(=), var, range), Expr(:block, body))
  end
end

# Detect if v1, v2 form a nested grid pattern, returning path info
function _detect_nesting(v1, v2, n, p1, p2)
  let inner_unique = unique(v2), m = length(inner_unique)
    if n % m == 0
      let k = n ÷ m
        if all(i -> v2[i] == inner_unique[((i-1) % m) + 1], 1:n)
          let outer_unique = [v1[(i-1)*m + 1] for i in 1:k]
            if all(i -> all(j -> v1[(i-1)*m + j] == outer_unique[i], 1:m), 1:k)
              return (p1, outer_unique, p2, inner_unique)
            end
          end
        end
      end
    end
    # Try swapped
    let inner_unique = unique(v1), m = length(inner_unique)
      if n % m == 0
        let k = n ÷ m
          if all(i -> v1[i] == inner_unique[((i-1) % m) + 1], 1:n)
            let outer_unique = [v2[(i-1)*m + 1] for i in 1:k]
              if all(i -> all(j -> v2[(i-1)*m + j] == outer_unique[i], 1:m), 1:k)
                return (p2, outer_unique, p1, inner_unique)
              end
            end
          end
        end
      end
    end
    nothing
  end
end

function _try_reroll_2d(template, paths, value_lists)
  let p1 = paths[1], p2 = paths[2],
      v1 = value_lists[p1], v2 = value_lists[p2],
      n = length(v1),
      nested = _detect_nesting(v1, v2, n, p1, p2)
    nested === nothing && return nothing
    let (outer_path, outer_vals, inner_path, inner_vals) = nested,
        outer_range = _try_make_range(outer_vals),
        inner_range = _try_make_range(inner_vals)
      (outer_range === nothing || inner_range === nothing) && return nothing
      let outer_var = gensym_short(:x),
          inner_var = gensym_short(:y),
          body = expr_replace_at(
                   expr_replace_at(template, outer_path, outer_var),
                   inner_path, inner_var)
        Expr(:for, Expr(:(=), outer_var, outer_range),
             Expr(:block,
                  Expr(:for, Expr(:(=), inner_var, inner_range),
                       Expr(:block, body))))
      end
    end
  end
end

gensym_short(base::Symbol) =
  let names = Dict(:x => :x, :y => :y, :z => :z, :i => :i, :j => :j)
    get(names, base, base)
  end

# Try to convert a value list to a range expression
function _try_make_range(vals)
  # Only numeric leaves can form a range/step; a non-numeric differing leaf (e.g. obj_model file paths,
  # or family names) means this group can't be rerolled — signal rejection so it emits individually.
  all(v -> v isa Real, vals) || return nothing
  n = length(vals)
  n < 2 && return Expr(:vect, vals...)
  sorted = sort(vals)
  spacing = sorted[2] - sorted[1]
  if spacing > 0 && all(i -> abs((sorted[i] - sorted[i-1]) - spacing) < 0.001, 2:n)
    let first_v = meta_program(sorted[1]),
        step_v = meta_program(spacing),
        last_v = meta_program(sorted[end])
      if step_v == 1
        Expr(:call, :(:), first_v, last_v)
      else
        # A float first:step:last range is NOT count-preserving — endpoint rounding can drop or add
        # an element (e.g. 0.1:0.1:0.3 yields 2 elements, not 3), so the rerolled loop would place
        # the wrong number of shapes. range(first; step, length=n) forces exactly the original n
        # elements while keeping the exact step. (step == 1 stays a:b — integer ranges are exact.)
        Expr(:call, :range, first_v, Expr(:kw, :step, step_v), Expr(:kw, :length, n))
      end
    end
  else
    Expr(:vect, map(meta_program, sorted)...)
  end
end

# ─── Shared structural helpers for the abstraction passes ─────────────────────

# Bool <: Real in Julia; a true/false leaf must never be treated as a reroll/parameter value.
_num(x) = x isa Real && !(x isa Bool)

# Structural equality modulo numeric leaves: same tree shape, non-numeric leaves equal.
_same_shape(a::Expr, b::Expr) =
  a.head == b.head && length(a.args) == length(b.args) &&
  all(_same_shape(x, y) for (x, y) in zip(a.args, b.args))
_same_shape(a, b) = (a isa Expr || b isa Expr) ? false : (a == b || (_num(a) && _num(b)))

# Fingerprint an Expr with numeric leaves wildcarded — statements with the same fingerprint are
# candidates for sorting groups and clone extraction.
_wildcard_reals(e) = map_expr(x -> _num(x) ? :■ : x, e)

# Numeric leaves in traversal order (the canonical sort key of a statement).
_numeric_leaves(x) = _num(x) ? Float64[Float64(x)] : Float64[]
_numeric_leaves(e::Expr) =
  let out = Float64[]
    for a in e.args
      append!(out, _numeric_leaves(a))
    end
    out
  end

_lex_less(a::Vector{Float64}, b::Vector{Float64}) =
  let n = min(length(a), length(b))
    for i in 1:n
      a[i] < b[i] && return true
      a[i] > b[i] && return false
    end
    length(a) < length(b)
  end

_is_level_assign(s) =
  s isa Expr && s.head == :(=) && s.args[2] isa Expr &&
  s.args[2].head == :call && s.args[2].args[1] == :level

# Includes unconnected-top markers (unconnected_level): they group and section with the levels,
# but parametrize_levels stays STRICT — a phantom top height must not break floor_height detection.
_is_any_level_assign(s) =
  s isa Expr && s.head == :(=) && s.args[2] isa Expr &&
  s.args[2].head == :call && s.args[2].args[1] in (:level, :unconnected_level)

# Statements the clone/repetition passes may lift into functions: plain element calls and
# wall let-blocks. Families, levels, groups, backend mappings, and mesh fallbacks stay put.
_clone_eligible(s) =
  s isa Expr &&
  (s.head == :let ||
   (s.head == :call && s.args[1] isa Symbol &&
    !(s.args[1] in _family_function_names) &&
    !(s.args[1] in (:level, :set_backend_family, :group, :group_instance, :obj_model,
                    :add_resource_folder!))))

# 3.4b Canonical statement order: introspection yields element-id (creation) order, so a perfect
# grid arrives shuffled and the consecutive-run reroller never fires. Within each run of
# statements sharing a shape (same head/callee/arity), group by non-numeric fingerprint (family
# variable, path shape, …) preserving first-occurrence group order, and sort each group by its
# numeric leaves. Reordering is confined to structurally-identical independent element statements,
# and also makes generation deterministic under Revit's unspecified element enumeration order.
_sortable_key(s) =
  s isa Expr ?
    (s.head == :call && s.args[1] isa Symbol && _clone_eligible(s) ?
       (:call, s.args[1], length(s.args)) :
     s.head == :let ? (:let,) : nothing) :
    nothing

_canonical_run_order(run) =
  length(run) < 2 ? run :
  let groups = Vector{Vector{Any}}(),
      fpstrs = String[],
      index = Dict{UInt, Int}()
    for s in run
      let fp = hash(_wildcard_reals(s)),
          gi = get!(index, fp) do
            push!(groups, Any[])
            push!(fpstrs, string(_wildcard_reals(s)))
            length(groups)
          end
        push!(groups[gi], s)
      end
    end
    # Canonical group order too (fingerprint text, then first member's numeric key): the arrival
    # order comes from the backend's unspecified enumeration and must not leak into the output.
    let sorted_groups = [sort(g, by=_numeric_leaves, lt=_lex_less, alg=MergeSort) for g in groups],
        perm = sortperm(collect(1:length(sorted_groups)),
                        lt=(a, b) -> fpstrs[a] != fpstrs[b] ? fpstrs[a] < fpstrs[b] :
                              _lex_less(_numeric_leaves(sorted_groups[a][1]),
                                        _numeric_leaves(sorted_groups[b][1])),
                        alg=MergeSort)
      reduce(vcat, sorted_groups[perm], init=Any[])
    end
  end

function sort_statements(e::Expr)
  stmts = collect(stmts_of(e))
  result = Any[]
  i = 1
  while i <= length(stmts)
    let k = _sortable_key(stmts[i])
      if k === nothing
        push!(result, stmts[i])
        i += 1
      else
        let j = i
          while j < length(stmts) && _sortable_key(stmts[j+1]) == k
            j += 1
          end
          append!(result, _canonical_run_order(stmts[i:j]))
          i = j + 1
        end
      end
    end
  end
  Expr(:block, result...)
end

# 3.5 Detect level repetition: identical element sets on successive levels (typical floors) are
# lifted into a `typical_floor(lvl_0, lvl_1, …)` function — called per floor, or looped over a
# `building_levels` vector when ≥3 consecutive floors match (making the floor count a loop bound).
_normalize_level_syms(s, base, order) =
  map_expr(x -> x isa Symbol && haskey(order, x) ? Symbol("__lvl_", order[x] - base) : x, s)

function detect_level_repetition(e::Expr)
  stmts = collect(stmts_of(e))
  level_syms = Symbol[s.args[1] for s in stmts if _is_any_level_assign(s)]
  length(level_syms) >= 3 || return e
  order = Dict{Symbol, Int}(sym => i for (i, sym) in enumerate(level_syms))
  # Bucket element statements by the lowest level they reference (their floor).
  buckets = Dict{Int, Vector{Int}}()
  for (pos, s) in enumerate(stmts)
    _clone_eligible(s) || continue
    let refs = collect_exprs(x -> x isa Symbol && haskey(order, x), s)
      isempty(refs) && continue
      push!(get!(buckets, minimum(order[r] for r in refs), Int[]), pos)
    end
  end
  # Group floors by the multiset of their level-normalized statements.
  bysig = Dict{Vector{UInt}, Vector{Int}}()
  for (base, poss) in buckets
    length(poss) >= 2 || continue
    let sg = sort!([hash(_normalize_level_syms(stmts[p], base, order)) for p in poss])
      push!(get!(bysig, sg, Int[]), base)
    end
  end
  consumed = falses(length(stmts))
  inserts = Dict{Int, Vector{Any}}()
  fcount = 0
  levels_vect_emitted = false
  for sg in sort!(collect(keys(bysig)), by=hash)
    bases = sort!(bysig[sg])
    length(bases) >= 2 || continue
    # Structural verification (hashes only bucket): every later floor must normalize to the same
    # statement multiset as the first — compare via sorted printed forms.
    let base1 = bases[1],
        norm_set = b -> sort!([string(_normalize_level_syms(stmts[p], b, order)) for p in buckets[b]])
      all(b -> norm_set(b) == norm_set(base1), bases[2:end]) || continue
    end
    fcount += 1
    let fname = fcount == 1 ? :typical_floor : Symbol("typical_floor_", fcount),
        base1 = bases[1],
        poss1 = buckets[base1],
        offs = sort!(unique!(reduce(vcat,
                 [Int[order[r] - base1
                      for r in collect_exprs(x -> x isa Symbol && haskey(order, x), stmts[p])]
                  for p in poss1],
                 init=Int[]))),
        params = [Symbol("lvl_", o) for o in offs],
        pmap = Dict(o => p for (o, p) in zip(offs, params)),
        body = [map_expr(x -> x isa Symbol && haskey(order, x) ? pmap[order[x] - base1] : x,
                         stmts[p])
                for p in poss1],
        def = Expr(:function, Expr(:call, fname, params...), Expr(:block, body...)),
        emit = Any[def],
        contiguous = bases == collect(bases[1]:bases[end]) && offs == collect(0:length(offs)-1)
      if length(bases) >= 3 && contiguous
        levels_vect_emitted ||
          push!(emit, Expr(:(=), :building_levels, Expr(:vect, level_syms...)))
        levels_vect_emitted = true
        push!(emit,
              Expr(:for, Expr(:(=), :i, Expr(:call, :(:), bases[1], bases[end])),
                   Expr(:block,
                        Expr(:call, fname,
                             [o == 0 ? Expr(:ref, :building_levels, :i) :
                                       Expr(:ref, :building_levels, Expr(:call, :+, :i, o))
                              for o in offs]...))))
      else
        append!(emit, [Expr(:call, fname, [level_syms[b + o] for o in offs]...) for b in bases])
      end
      inserts[minimum(minimum(buckets[b]) for b in bases)] = emit
      for b in bases, p in buckets[b]
        consumed[p] = true
      end
    end
  end
  any(consumed) || return e
  let out = Any[]
    for (i, s) in enumerate(stmts)
      haskey(inserts, i) && append!(out, inserts[i])
      consumed[i] || push!(out, s)
    end
    Expr(:block, out...)
  end
end

# 3.5b Function extraction (clone detection + anti-unification): repeated statement subsequences
# whose occurrences differ only in numeric leaves become a parameterized function plus one call per
# occurrence. When the variation is a pure translation of xy/xyz coordinates, the parameter list
# collapses to a single placement point `p` and the body is rewritten origin-relative — the same
# idiom as the Revit-group factory emission. Call sites are plain calls, so a run of them can then
# reroll into a for-loop downstream.

# The coordinate slot of a varying leaf: (axis, :xy|:xyz) when the leaf is an argument of an
# xy/xyz call, nothing otherwise.
_coord_axis(tpl, path) =
  length(path) < 2 ? nothing :
  let parent = _get_diff_value(tpl, path[1:end-1])
    parent isa Expr && parent.head == :call && parent.args[1] in (:xy, :xyz) && path[end] >= 2 ?
      (path[end] - 1, parent.args[1]) : nothing
  end

_seqs_same_shape(stmts, s1, s2, L) =
  all(k -> _same_shape(stmts[s1+k-1], stmts[s2+k-1]), 1:L)

# Build (def, Dict start=>call, Dict start=>span) for a clone, or nothing when the variation is
# not abstractable (non-numeric leaves, or too many scalar parameters to be readable).
function _clone_function(stmts, sel, L, counter; max_params=6)
  occs = [stmts[s:s+L-1] for s in sel]
  tpl = occs[1]
  var_paths = [Vector{Vector{Int}}() for _ in 1:L]
  for occ in occs[2:end], k in 1:L
    for (p, v1, v2) in expr_diff(tpl[k], occ[k])
      (_num(v1) && _num(v2)) || return nothing
      p in var_paths[k] || push!(var_paths[k], p)
    end
  end
  foreach(sort!, var_paths)
  flat = [(k, p) for k in 1:L for p in var_paths[k]]
  vals = [Any[_get_diff_value(occ[k], p) for (k, p) in flat] for occ in occs]
  base_name = let s0 = tpl[1]
    s0.head == :let ? s0.args[1].args[2].args[1] : s0.args[1]
  end
  # Number only ACCEPTED clones (rejected candidates must not burn counter values).
  mkname = () -> Symbol(base_name, :_cluster_, counter[] += 1)
  axinfos = [_coord_axis(tpl[k], p) for (k, p) in flat]
  parents = unique([(k, p[1:end-1]) for (k, p) in flat])
  # Translation-collapse: possible when every varying leaf is a coordinate of one kind of call
  # (all :xy or all :xyz), every such parent call has all-numeric coordinates, and the
  # per-occurrence deltas are consistent per axis.
  collapsible = !isempty(flat) && all(!isnothing, axinfos) &&
    length(unique(map(last, axinfos))) == 1 &&
    all(p -> let c = _get_diff_value(tpl[p[1]], p[2])
          all(_num, c.args[2:end])
        end, parents)
  if collapsible
    let kind = last(axinfos[1]),
        base_vals = vals[1],
        deltas = Vector{NTuple{3, Float64}}(),
        ok = true
      for vj in vals
        let d = Dict{Int, Float64}()
          for (m, (ax, _)) in enumerate(axinfos)
            let dm = Float64(vj[m] - base_vals[m])
              if haskey(d, ax) && abs(d[ax] - dm) > 1e-6
                ok = false
                break
              end
              d[ax] = dm
            end
          end
          ok || break
          push!(deltas, (get(d, 1, 0.0), get(d, 2, 0.0), get(d, 3, 0.0)))
        end
      end
      if ok
        # Anchor = the parent coordinate call of the first varying leaf; body: anchor → p,
        # every other varying coordinate call → p + vxy/vxyz(offset-from-anchor).
        let (k0, p0) = flat[1],
            anchor_path = p0[1:end-1],
            anchor = _get_diff_value(tpl[k0], anchor_path),
            a0 = let aa = map(Float64, anchor.args[2:end])
              (aa[1], aa[2], length(aa) >= 3 ? aa[3] : 0.0)
            end,
            body = Any[]
          for k in 1:L
            let bk = tpl[k]
              # Replace deeper parents first so shallower paths stay valid (paths are into tpl[k]
              # and replacing a whole parent call cannot invalidate a disjoint parent's path).
              for (kk, pp) in sort([pr for pr in parents if pr[1] == k], by=pr -> -length(pr[2]))
                let c = _get_diff_value(tpl[k], pp)
                  bk = (kk, pp) == (k0, anchor_path) ?
                    expr_replace_at(bk, pp, :p) :
                    let cargs = map(Float64, c.args[2:end]),
                        rel = [meta_program(cargs[i] - (a0[i])) for i in 1:length(cargs)]
                      expr_replace_at(bk, pp,
                        all(r -> r == 0 || r == 0.0, rel) ? :p :
                          Expr(:call, :+, :p, Expr(:call, kind == :xy ? :vxy : :vxyz, rel...)))
                    end
                end
              end
              push!(body, bk)
            end
          end
          let fname = mkname(),
              def = Expr(:function, Expr(:call, fname, :p), Expr(:block, body...)),
              callmap = Dict{Int, Any}(),
              spanmap = Dict{Int, Int}()
            for (j, st) in enumerate(sel)
              let (dx, dy, dz) = deltas[j],
                  coords = kind == :xy ?
                    [meta_program(a0[1] + dx), meta_program(a0[2] + dy)] :
                    [meta_program(a0[1] + dx), meta_program(a0[2] + dy), meta_program(a0[3] + dz)]
                callmap[st] = Expr(:call, fname, Expr(:call, kind, coords...))
                spanmap[st] = L
              end
            end
            return (def, callmap, spanmap)
          end
        end
      end
    end
  end
  # General scalar parameterization.
  length(flat) <= max_params || return nothing
  let params = [Symbol("p", m) for m in 1:length(flat)],
      body = Any[]
    for k in 1:L
      let bk = tpl[k]
        for (m, (kk, pp)) in enumerate(flat)
          kk == k && (bk = expr_replace_at(bk, pp, params[m]))
        end
        push!(body, bk)
      end
    end
    let fname = mkname(),
        def = Expr(:function, Expr(:call, fname, params...), Expr(:block, body...)),
        callmap = Dict{Int, Any}(),
        spanmap = Dict{Int, Int}()
      for (j, st) in enumerate(sel)
        callmap[st] = Expr(:call, fname, map(meta_program, vals[j])...)
        spanmap[st] = L
      end
      (def, callmap, spanmap)
    end
  end
end

function extract_functions(e::Expr; min_stmts=2, min_occ=2, min_saving=1)
  stmts = collect(stmts_of(e))
  n = length(stmts)
  fps = Union{UInt, Nothing}[_clone_eligible(s) ? hash(_wildcard_reals(s)) : nothing
                             for s in stmts]
  consumed = falses(n)
  defs = Dict{Int, Vector{Any}}()
  calls = Dict{Int, Any}()
  spans = Dict{Int, Int}()
  counter = Ref(0)
  for L in (n ÷ min_occ):-1:min_stmts
    let table = Dict{UInt, Vector{Int}}()
      for i in 1:(n - L + 1)
        all(j -> fps[i+j-1] !== nothing && !consumed[i+j-1], 1:L) || continue
        # A window of one repeated fingerprint is a uniform run — loop_rerolling's domain, and
        # extracting an arbitrary slice of it (e.g. half a column grid) is worse than the loop.
        # Clones must be heterogeneous patterns (wall+slab, wall-with-door+wall, …).
        length(unique(view(fps, i:i+L-1))) >= 2 || continue
        push!(get!(table, hash(fps[i:i+L-1]), Int[]), i)
      end
      for (key, starts) in sort!(collect(table), by=kv -> first(kv[2]))
        length(starts) >= min_occ || continue
        let sel = Int[]
          for st in starts
            (isempty(sel) || st > sel[end] + L - 1) || continue
            all(j -> !consumed[st+j-1], 1:L) || continue
            (isempty(sel) || _seqs_same_shape(stmts, sel[1], st, L)) || continue
            push!(sel, st)
          end
          length(sel) >= min_occ || continue
          L * length(sel) - (L + length(sel)) >= min_saving || continue
          let cf = _clone_function(stmts, sel, L, counter)
            cf === nothing && continue
            let (def, callmap, spanmap) = cf
              push!(get!(defs, sel[1], Any[]), def)
              merge!(calls, callmap)
              merge!(spans, spanmap)
              for st in sel
                consumed[st:st+L-1] .= true
              end
            end
          end
        end
      end
    end
  end
  any(consumed) || return e
  let out = Any[], i = 1
    while i <= n
      haskey(defs, i) && append!(out, defs[i])
      if haskey(calls, i)
        push!(out, calls[i])
        i += spans[i]
      elseif consumed[i]
        i += 1
      else
        push!(out, stmts[i])
        i += 1
      end
    end
    Expr(:block, out...)
  end
end

# ─── Parametrization passes (Goal 3) ─────────────────────────────────────────
# Turn the literals a user would want to edit — floor heights, grid counts and spacings —
# into named globals at the top of the generated program, initialized from the model.

# 3.7a Uniform level heights → base_level_height / floor_height / n_levels parameters.
function parametrize_levels(e::Expr)
  stmts = collect(stmts_of(e))
  li = [i for (i, s) in enumerate(stmts) if _is_level_assign(s)]
  length(li) >= 3 || return e
  hs = [stmts[i].args[2].args[2] for i in li]
  all(_num, hs) || return e
  issorted(hs) || return e
  let d = hs[2] - hs[1]
    d > 1e-6 || return e
    all(k -> abs((hs[k] - hs[k-1]) - d) <= 1e-3, 2:length(hs)) || return e
    let base = hs[1],
        has_base = abs(base) > 1e-9,
        params = Any[],
        hexpr = k ->   # k is 0-based level index
          k == 0 ? (has_base ? :base_level_height : meta_program(0.0)) :
          let step = k == 1 ? :floor_height : Expr(:call, :*, k, :floor_height)
            has_base ? Expr(:call, :+, :base_level_height, step) : step
          end,
        out = collect(stmts)
      has_base && push!(params, Expr(:(=), :base_level_height, meta_program(base)))
      push!(params, Expr(:(=), :floor_height, meta_program(d)))
      push!(params, Expr(:(=), :n_levels, length(hs)))
      for (k, i) in enumerate(li)
        out[i] = Expr(:(=), stmts[i].args[1], Expr(:call, :level, hexpr(k - 1)))
      end
      # Also express unconnected TOP levels (wall-top markers, e.g. a roof datum) as a multiple of
      # floor_height plus a remainder, matching the regular levels' form — they were previously left
      # as a raw literal, inconsistent with level_1/level_2. Value-preserving within meta_program's
      # 8-sigdigit rounding (~1e-8; floor_height and the remainder are each rounded when printed),
      # the same tolerance class as the regular level parametrization above.
      # Decompose only datums that plausibly TRACK the storey stack: k within the stack (plus
      # one storey above the top) and a small rise/drop (|r| <= d/4). A mid-storey datum
      # (h=4.9 in a 3.26 grid) or a far-off site datum (h=100) would otherwise get a
      # floor_height coefficient that misstates design intent — those stay literal. Negative
      # remainders emit as a subtraction, not "+ -0.52".
      for (i, s) in enumerate(stmts)
        (_is_any_level_assign(s) && !_is_level_assign(s) && s.args[2].args[2] isa Real) || continue
        let h = Float64(s.args[2].args[2]), k = round(Int, (h - base) / d)
          if 1 <= k <= length(hs) && abs(h - base - k * d) <= d / 4
            let r = meta_program(h - base - k * d),
                mult = k == 1 ? :floor_height : Expr(:call, :*, k, :floor_height),
                withbase = has_base ? Expr(:call, :+, :base_level_height, mult) : mult,
                rhs = r == 0 ? withbase :
                      r > 0 ? Expr(:call, :+, withbase, r) :
                              Expr(:call, :-, withbase, meta_program(-r))
              out[i] = Expr(:(=), s.args[1], Expr(:call, :unconnected_level, rhs))
            end
          end
        end
      end
      Expr(:block, params..., out...)
    end
  end
end

# 3.7b Rerolled loop ranges → count + spacing parameters (n_columns_x, columns_x_spacing, …).
_loop_body_callee(body) =
  let stmts = stmts_of(body)
    for s in stmts
      s isa Expr || continue
      s.head == :for && return _loop_body_callee(s.args[2])
      s.head == :call && return s.args[1]
      s.head == :let && return s.args[1].args[2].args[1]
    end
    nothing
  end

_loop_var_axis(body, var) =
  let hits = collect_exprs(
        x -> x isa Expr && x.head == :call && x.args[1] in (:xy, :xyz) && var in x.args, body)
    length(hits) == 1 ? ("xyz"[findfirst(==(var), hits[1].args) - 1]) : nothing
  end

_fresh_param(base, used) =
  let sym = Symbol(base), k = 2
    while sym in used
      sym = Symbol(base, "_", k)
      k += 1
    end
    push!(used, sym)
    sym
  end

_symbolize_for(s, params, used) = s
_symbolize_for(s::Expr, params, used) =
  s.head == :for ?
    let binding = s.args[1],
        var = binding.args[1],
        rng = binding.args[2],
        # Symbolize THIS loop's range before recursing so the emitted parameter
        # order follows the loop nesting (n_columns_x before n_columns_y), the
        # way an author writes the knobs.
        newbinding =
          if rng isa Expr && rng.head == :call && rng.args[1] == :range
            let step_kw = findfirst(a -> a isa Expr && a.head == :kw && a.args[1] == :step, rng.args),
                len_kw = findfirst(a -> a isa Expr && a.head == :kw && a.args[1] == :length, rng.args)
              if step_kw !== nothing && len_kw !== nothing
                let callee = _loop_body_callee(s.args[2]),
                    axis = _loop_var_axis(s.args[2], var),
                    base = callee === nothing ? "items" :
                           let cs = string(callee)
                             endswith(cs, "s") ? cs : cs * "s"
                           end,
                    suffix = axis === nothing ? "" : "_$(axis)",
                    nsym = _fresh_param("n_$(base)$(suffix)", used),
                    ssym = _fresh_param("$(base)$(suffix)_spacing", used)
                  push!(params, Expr(:(=), ssym, rng.args[step_kw].args[2]))
                  push!(params, Expr(:(=), nsym, rng.args[len_kw].args[2]))
                  Expr(:(=), var,
                       Expr(:call, :range, rng.args[2],
                            Expr(:kw, :step, ssym), Expr(:kw, :length, nsym)))
                end
              else
                binding
              end
            end
          else
            binding
          end,
        body = Expr(:block, [_symbolize_for(b, params, used) for b in stmts_of(s.args[2])]...)
      Expr(:for, newbinding, body)
    end :
  s

function symbolize_ranges(e::Expr)
  stmts = collect(stmts_of(e))
  params = Any[]
  used = Set{Symbol}(s.args[1] for s in stmts
                     if s isa Expr && s.head == :(=) && s.args[1] isa Symbol)
  out = Any[_symbolize_for(s, params, used) for s in stmts]
  isempty(params) && return e
  Expr(:block, params..., out...)
end

# 3.7c Conservative constant unification: a literal that exactly matches an already-introduced
# distinctive parameter value is replaced by the parameter, so dimensional relations surface.
# "Distinctive" excludes round numbers (5.0, 0.2) whose equality is likely coincidence.
_distinctive(v) = abs(v - round(v, digits=1)) > 5e-4

_unify_expr(x, table) =
  x isa AbstractFloat && haskey(table, Float64(x)) ? table[Float64(x)] : x
_unify_expr(e::Expr, table) =
  (e.head == :call && e.args[1] isa Symbol && _protected_callee(e.args[1])) ?
    e :
    Expr(e.head, [_unify_expr(a, table) for a in e.args]...)

function unify_constants(e::Expr)
  stmts = collect(stmts_of(e))
  table = Dict{Float64, Symbol}()
  ambiguous = Set{Float64}()
  for s in stmts
    (s isa Expr && s.head == :(=) && s.args[1] isa Symbol && s.args[2] isa AbstractFloat) ||
      continue
    let v = Float64(s.args[2])
      _distinctive(v) || continue
      haskey(table, v) ? push!(ambiguous, v) : (table[v] = s.args[1])
    end
  end
  for v in ambiguous
    delete!(table, v)
  end
  isempty(table) && return e
  Expr(:block,
       [s isa Expr && s.head == :(=) && s.args[2] isa AbstractFloat ? s : _unify_expr(s, table)
        for s in stmts]...)
end

# 3.7d Symbolic angles — element rotation arguments are emitted as raw radian floats
# (1.5707963, 3.1415927, …). Rewrite those that are simple π-fractions into readable `pi` /
# `pi / 2` / `-(pi / 2)` / … . Coverage: the angle-bearing POSITIONAL of family_element and
# column (both at e.args[3], i.e. the 2nd positional), and the `angle=` KEYWORD of beam and
# free_column (their meta_program emission form). The symbolic value differs from the
# 8-sigdigit literal by ~1e-8 rad — far inside the oracle's 2e-3 tolerance — so geometry is
# preserved while the intent (quarter/half/full turn) becomes explicit.
const _angle_arg_positions = Dict{Symbol, Int}(:family_element => 3, :column => 3)
const _angle_kw_callees = Set([:beam, :free_column])

_pi_frac_expr(n::Int, d::Int) =
  let g = gcd(abs(n), d), n2 = n ÷ g, d2 = d ÷ g,
      mag = d2 == 1 ?
              (abs(n2) == 1 ? :pi : Expr(:call, :*, abs(n2), :pi)) :
              (abs(n2) == 1 ? Expr(:call, :/, :pi, d2) :
                              Expr(:call, :/, Expr(:call, :*, abs(n2), :pi), d2))
    n2 < 0 ? Expr(:call, :-, mag) : mag
  end

#=
Recognize v ≈ (n/d)·π for a small denominator; return the symbolic Expr or nothing (incl.
v ≈ 0). The tolerance is just enough to absorb meta_program's sigdigit rounding of an exact
π-fraction (~5e-8 at the default 8 sigdigits for |v| ≤ 2π) — NOT loose enough to swallow
genuinely non-π survey angles: the old 1e-5 window rewrote a 59.99944° fixture rotation
(1.0471879, off π/3 by 9.7e-6) to `pi / 3`, silently erasing surveyed orientation intent.
=#
_angle_symbol(v::Real) =
  abs(v) < 1e-5 ? nothing :
  let tol = 10.0^(1 - meta_program_sigdigits()) * 2π,
      hit = nothing
    for d in (1, 2, 3, 4, 6)
      let n = round(Int, v * d / pi)
        if n != 0 && abs(v - n * pi / d) < tol
          hit = (n, d)
          break
        end
      end
    end
    hit === nothing ? nothing : _pi_frac_expr(hit[1], hit[2])
  end

_symbolize_angle_call(x) =
  let x2 = if haskey(_angle_arg_positions, x.args[1]) &&
              length(x.args) >= _angle_arg_positions[x.args[1]] &&
              _num(x.args[_angle_arg_positions[x.args[1]]])
        let i = _angle_arg_positions[x.args[1]], s = _angle_symbol(x.args[i])
          s === nothing ? x : Expr(x.head, x.args[1:i-1]..., s, x.args[i+1:end]...)
        end
      else
        x
      end
    x2.args[1] in _angle_kw_callees ?
      Expr(:call,
           [a isa Expr && a.head == :kw && a.args[1] == :angle && _num(a.args[2]) &&
              _angle_symbol(a.args[2]) !== nothing ?
              Expr(:kw, :angle, _angle_symbol(a.args[2])) : a
            for a in x2.args]...) :
      x2
  end

symbolize_angles(e::Expr) =
  map_expr(x -> x isa Expr && x.head == :call && x.args[1] isa Symbol ?
                  _symbolize_angle_call(x) : x, e)

# 3.7e Extract shared dimensions — recurring numeric KEYWORD-argument values (a wall's top/base
# offset, …) are one physical dimension repeated across many elements. Lift each value that recurs
# at least `_dim_min_occurrences` times under a single (callee, keyword) context into a named
# parameter `<callee>_<keyword>` and replace its occurrences, so the shared dimension becomes one
# editable knob. Float kwargs only — an Int kwarg is a count, not a dimension, and lifting it as a
# Float64 param silently changes call-site types (range(length=7.0) throws).
#
# Counting is by ELEMENT, not by statement: a statement inside a rerolled for-loop counts once per
# iteration (statically-known range length), and a pass-emitted function body counts once per call
# site — otherwise the dimension shared by the MOST repetitive geometry (8 rerolled parapet walls)
# never reaches the threshold while 4 flat walls elsewhere do, splitting one physical dimension.
# Values within the reroll snap class (1e-3) merge into one knob whose value is the
# heaviest-weighted member (introspection wobble like -0.19999999 joins -0.2's knob; the ≤1e-3
# shift is the same tolerance class the reroll passes already introduce).
const _dim_min_occurrences = 4
const _dim_snap_tolerance = 1e-3

# The static multiplicity context: Int-valued prologue constants (for resolving symbolic range
# lengths) and per-function call counts (for weighing pass-emitted def bodies).
function _dim_weight_context(e::Expr)
  stmts = collect(stmts_of(e))
  consts = Dict{Symbol, Int}(s.args[1] => s.args[2] for s in stmts
                             if s isa Expr && s.head == :(=) && s.args[1] isa Symbol &&
                                s.args[2] isa Int)
  callcounts = Dict{Symbol, Int}()
  for c in collect_exprs(x -> x isa Expr && x.head == :call && x.args[1] isa Symbol, e)
    callcounts[c.args[1]] = get(callcounts, c.args[1], 0) + 1
  end
  (consts = consts, callcounts = callcounts)
end

_static_range_length(rng, consts) =
  rng isa Expr && rng.head == :call && rng.args[1] == :range ?
    let i = findfirst(a -> a isa Expr && a.head == :kw && a.args[1] == :length, rng.args)
      i === nothing ? nothing :
        let v = rng.args[i].args[2]
          v isa Int ? v : v isa Symbol ? get(consts, v, nothing) : nothing
        end
    end :
  rng isa Expr && rng.head == :call && rng.args[1] == :(:) && length(rng.args) == 3 &&
  rng.args[2] isa Int && rng.args[3] isa Int ?
    max(rng.args[3] - rng.args[2] + 1, 0) :
    nothing

# The multiplicity a subtree contributes with: loops multiply by their static length, function
# bodies by their call count (a def that is never called still counts once — its dimensions are
# real even if latent).
_stmt_weight(s, ctx) =
  s isa Expr && s.head == :for ?
    something(_static_range_length(s.args[1].args[2], ctx.consts), 1) :
  s isa Expr && s.head == :function && s.args[1] isa Expr && s.args[1].head == :call ?
    max(get(ctx.callcounts, s.args[1].args[1], 1), 1) :
    1

# Merge (value, weight) pairs whose values lie within the snap tolerance; each bucket keeps its
# member values (all get replaced) and takes the heaviest member as the representative.
function _snap_buckets(pairs)
  sorted = sort(pairs, by = first)
  buckets = Vector{Vector{Tuple{Float64, Int}}}()
  for (v, c) in sorted
    if !isempty(buckets) && v - buckets[end][end][1] <= _dim_snap_tolerance
      push!(buckets[end], (v, c))
    else
      push!(buckets, [(v, c)])
    end
  end
  [(rep = argmax(p -> p[2], b)[1], total = sum(p[2] for p in b),
    members = [p[1] for p in b]) for b in buckets]
end

# The kwargs of a call, whether direct `Expr(:kw, ...)` args or children of an
# `Expr(:parameters, ...)` block (pbr_material and group emit the latter form).
_call_kwargs(e::Expr) =
  [kw for a in e.args[2:end]
      for kw in (a isa Expr && a.head == :parameters ? a.args : (a,))
      if kw isa Expr && kw.head == :kw]

# Count (callee, kw-key, value) over element calls, never descending into protected constructs
# (the shared _protected_callee set), weighted by static multiplicity.
_collect_kw_dims(x, counts, mult, ctx) = nothing
function _collect_kw_dims(e::Expr, counts, mult, ctx)
  (e.head == :call && e.args[1] isa Symbol && _protected_callee(e.args[1])) && return nothing
  if e.head == :call && e.args[1] isa Symbol
    for a in _call_kwargs(e)
      # :angle kwargs are rotations, not dimensions — symbolize_angles (which runs first)
      # owns them; mining a recurring angle as "<callee>_angle = 1.5707963" would freeze a
      # rotation into a pseudo-length knob.
      if a.args[1] != :angle && a.args[2] isa AbstractFloat
        let k = (e.args[1], a.args[1], Float64(a.args[2]))
          counts[k] = get(counts, k, 0) + mult
        end
      end
    end
  end
  let m = mult * _stmt_weight(e, ctx)
    for a in e.args
      _collect_kw_dims(a, counts, m, ctx)
    end
  end
  nothing
end

# Replace matching kw values with their param symbol; symmetric skip so protected subtrees
# untouched; handles both direct kw args and :parameters blocks.
_replace_kw_dim_arg(callee, a, table) =
  a isa Expr && a.head == :kw && a.args[2] isa AbstractFloat &&
    haskey(table, (callee, a.args[1], Float64(a.args[2]))) ?
    Expr(:kw, a.args[1], table[(callee, a.args[1], Float64(a.args[2]))]) :
    _replace_kw_dims(a, table)
_replace_kw_dims(x, table) = x
_replace_kw_dims(e::Expr, table) =
  (e.head == :call && e.args[1] isa Symbol && _protected_callee(e.args[1])) ? e :
  e.head == :call && e.args[1] isa Symbol ?
    Expr(:call, e.args[1],
         [a isa Expr && a.head == :parameters ?
            Expr(:parameters, [_replace_kw_dim_arg(e.args[1], kw, table) for kw in a.args]...) :
            _replace_kw_dim_arg(e.args[1], a, table)
          for a in e.args[2:end]]...) :
    Expr(e.head, [_replace_kw_dims(a, table) for a in e.args]...)

function extract_shared_dimensions(e::Expr)
  counts = Dict{Tuple{Symbol, Symbol, Float64}, Int}()
  _collect_kw_dims(e, counts, 1, _dim_weight_context(e))
  bykey = Dict{Tuple{Symbol, Symbol}, Vector{Tuple{Float64, Int}}}()
  for ((callee, key, v), c) in counts
    v == 0.0 && continue
    push!(get!(bykey, (callee, key), Tuple{Float64, Int}[]), (v, c))
  end
  used = Set{Symbol}(s.args[1] for s in stmts_of(e)
                     if s isa Expr && s.head == :(=) && s.args[1] isa Symbol)
  table = Dict{Tuple{Symbol, Symbol, Float64}, Symbol}()
  assigns = Expr[]
  for (callee, key) in sort(collect(keys(bykey)))                 # deterministic order
    let kept = [b for b in _snap_buckets(bykey[(callee, key)]) if b.total >= _dim_min_occurrences]
      for (i, b) in enumerate(sort(kept, by = b -> (-b.total, b.rep)))
        let name = _fresh_param(i == 1 ? "$(callee)_$(key)" : "$(callee)_$(key)_$(i)", used)
          push!(assigns, Expr(:(=), name, b.rep))
          for v in b.members
            table[(callee, key, v)] = name
          end
        end
      end
    end
  end
  isempty(assigns) && return e
  Expr(:block, assigns..., stmts_of(_replace_kw_dims(e, table))...)
end

# 3.8 Optional: wrap the whole program in `function building(; params...) … end; building()` so the
# parameters become re-invocable keyword arguments. Off by default (generate_khepri_code kwarg).
function wrap_program_in_function(e::Expr; name=:building)
  stmts = collect(stmts_of(e))
  # Rounded params carry their provenance comment on an :inline_comment wrapper — the kwarg
  # form has nowhere for a comment, so unwrap (the rounded value survives, the comment doesn't).
  unwrap = s -> s isa Expr && s.head == :inline_comment ? s.args[1] : s
  isparam = s -> let u = unwrap(s)
    u isa Expr && u.head == :(=) && u.args[1] isa Symbol && u.args[2] isa Real
  end
  k = something(findfirst(s -> !isparam(s), stmts), length(stmts) + 1) - 1
  k == 0 && return e
  Expr(:block,
       Expr(:function,
            Expr(:call, name,
                 Expr(:parameters,
                      [Expr(:kw, unwrap(s).args[1], unwrap(s).args[2]) for s in stmts[1:k]]...)),
            Expr(:block, stmts[k+1:end]...)),
       Expr(:call, name))
end

# 3.9 Sectionalize by storey — organize the flat element statements the way an AD
# developer would write them: one named function per storey (real Revit storey names
# when the introspection provides them), called bottom-up. Runs after unify_constants
# (earlier passes iterate top-level statements only and must not find them wrapped)
# and before wrap/header.
#
# Partition rules:
# - Prologue (constant/level/family assigns, @isdefined guards) keeps its position.
# - :call/:let/:for statements referencing level symbols go to the storey of the
#   LOWEST level they reference (the same key detect_level_repetition buckets by);
#   unconnected_level markers don't own storeys (they're wall-top heights).
# - group_instance calls section by their group FACTORY's lowest referenced level
#   (instances only depend on globally-defined factories, so this is order-safe).
#   Factory/typical_floor function defs stay global.
# - Statements with no level reference (obj_model fallbacks, finalize_groups,
#   building_levels loops) keep their global position, so program semantics and
#   sort_statements' canonical intra-run order are preserved exactly.
# Sectioning is skipped for programs with fewer than 2 real storeys.

_static_eval(x::Real, _) = Float64(x)
_static_eval(x::Symbol, consts) = get(consts, x, nothing)
_static_eval(x::Expr, consts) =
  if x.head == :call && length(x.args) >= 3 && x.args[1] in (:+, :-, :*, :/)
    let vals = [_static_eval(a, consts) for a in x.args[2:end]]
      any(isnothing, vals) ? nothing :
        foldl((a, b) -> Base.invokelatest(getfield(Base, x.args[1]), a, b), vals)
    end
  else
    nothing
  end
_static_eval(x, _) = nothing

_storey_fn_name(name, taken) =
  let base = Symbol(sanitize_name(name)),
      fn = base in taken ? Symbol(string(base, "_storey")) : base
    while fn in taken
      fn = Symbol(string(fn, "_"))
    end
    fn
  end

sectionalize_by_storey(level_names::Dict{Float64, String}=Dict{Float64, String}()) =
  (e::Expr) -> _sectionalize_by_storey(e, level_names)

function _sectionalize_by_storey(e::Expr, level_names)
  stmts = collect(stmts_of(e))
  consts = Dict{Symbol, Float64}()
  level_heights = Dict{Symbol, Float64}()   # REAL storey levels only
  level_syms = Set{Symbol}()                # incl. unconnected markers
  factories = Dict{Symbol, Expr}()          # factory fn name => body
  group_factory = Dict{Symbol, Symbol}()    # group var => factory fn name
  taken = Set{Symbol}()
  for s in stmts
    if s isa Expr && s.head == :(=) && s.args[1] isa Symbol
      push!(taken, s.args[1])
      let v = s.args[1], rhs = s.args[2]
        if rhs isa Real
          consts[v] = Float64(rhs)
        elseif rhs isa Expr && rhs.head == :call && length(rhs.args) >= 2
          if rhs.args[1] == :level
            let h = _static_eval(rhs.args[2], consts)
              h === nothing || (level_heights[v] = h)
            end
            push!(level_syms, v)
          elseif rhs.args[1] == :unconnected_level
            push!(level_syms, v)
          elseif rhs.args[1] == :group
            for a in rhs.args
              a isa Expr && a.head == :parameters && for kw in a.args
                kw isa Expr && kw.head == :kw && kw.args[1] == :factory &&
                  kw.args[2] isa Symbol && (group_factory[v] = kw.args[2])
              end
              a isa Expr && a.head == :kw && a.args[1] == :factory &&
                a.args[2] isa Symbol && (group_factory[v] = a.args[2])
            end
          end
        end
      end
    elseif s isa Expr && s.head == :function && s.args[1] isa Expr &&
           s.args[1].head == :call && s.args[1].args[1] isa Symbol
      push!(taken, s.args[1].args[1])
      factories[s.args[1].args[1]] = s
    end
  end
  length(level_heights) >= 2 || return e
  # Storey functions must not shadow anything the program CALLS (a storey named
  # "Roof" would define function roof() and break every roof(...) statement) —
  # seed the taken set with every call-position symbol in the program.
  for c in collect_exprs(x -> x isa Expr && x.head == :call && x.args[1] isa Symbol, e)
    push!(taken, c.args[1])
  end

  # storey key = the level var with the LOWEST height among those referenced
  storey_of_syms(syms) =
    let real = [s for s in syms if haskey(level_heights, s)]
      isempty(real) ? nothing : argmin(s -> level_heights[s], real)
    end
  refs_in(x) = collect_exprs(y -> y isa Symbol && y in level_syms, x)
  storey_of(s) =
    if s isa Expr && s.head == :call && s.args[1] == :group_instance &&
       length(s.args) >= 2 && s.args[2] isa Symbol
      let f = get(group_factory, s.args[2], nothing),
          body = f === nothing ? nothing : get(factories, f, nothing)
        body === nothing ? nothing : storey_of_syms(refs_in(body))
      end
    elseif s isa Expr && s.head in (:call, :let, :for)
      storey_of_syms(refs_in(s))
    else
      nothing
    end

  sections = Dict{Symbol, Vector{Any}}()
  out = Any[]
  first_section_at = nothing
  for s in stmts
    let key = s isa Expr && s.head == :function ? nothing : storey_of(s)
      if key === nothing
        push!(out, s)
      else
        push!(get!(sections, key, Any[]), s)
        first_section_at === nothing && (first_section_at = length(out) + 1)
      end
    end
  end
  isempty(sections) && return e

  ordered = sort(collect(keys(sections)), by=v -> level_heights[v])
  fnames = Dict{Symbol, Symbol}()
  for v in ordered
    let name = let h = level_heights[v],
                   k = findfirst(kh -> abs(kh - h) < 0.05, collect(keys(level_names)))
                 k === nothing ? "storey_" * string(v)[7:end] :
                                 level_names[collect(keys(level_names))[k]]
               end
      fnames[v] = _storey_fn_name(name, taken)
      push!(taken, fnames[v])
    end
  end
  #=
  ONE building origin, the exemplar AD idiom (Isenberg's b_center, Astana's
  astana_center_p): the SW corner of the literal-coordinate geometry across ALL
  storeys and pass-emitted global function bodies becomes a single
  `building_origin = xy(gx, gy)` parameter. Every storey function takes its
  placement anchor as `p0=building_origin` and its body is rewritten relative
  to p0 by the TOTAL rewriter (loop-variable and parameter-symbol coordinates
  rebase symbolically — see _anchor_relative_expr). Pass-emitted global
  functions that construct world geometry (typical_floor, clusters) gain a
  trailing `p0=building_origin` parameter, their bodies are rebased the same
  way, and storey-body call sites forward the storey's p0 — so storey_k()
  reproduces the original, storey_k(x(50)) coherently re-sites one storey, and
  editing building_origin moves the whole building. Group FACTORY bodies are
  excluded (already instance-relative; their group_instance placements rebase
  instead). World positions carry Revit's survey offset, which cancels in
  (coord - origin), so the emitted deltas are clean plan dimensions.
  =#
  factory_names = Set{Symbol}(values(group_factory))
  all_coords = Tuple{Float64, Float64}[]
  for v in ordered, s in sections[v]
    _collect_world_xy(s, all_coords)
  end
  rebased_def_names = Set{Symbol}()
  for (name, def) in factories
    name in factory_names && continue
    let acc = Tuple{Float64, Float64}[]
      _collect_world_xy(def.args[2], acc)
      if !isempty(acc)
        push!(rebased_def_names, name)
        append!(all_coords, acc)
      end
    end
  end
  defs = Any[]
  origin_stmts = Any[]
  if isempty(all_coords)
    # Nothing anchors: keep plain zero-argument storey functions.
    for v in ordered
      push!(defs, Expr(:toplevel_comment,
                       "Storey $(fnames[v]) ($(v) = $(round(level_heights[v], digits=3)))"))
      push!(defs, Expr(:function, Expr(:call, fnames[v]), Expr(:block, sections[v]...)))
    end
  else
    let gx = meta_program(minimum(first.(all_coords))),
        gy = meta_program(minimum(last.(all_coords))),
        origin = _fresh_param("building_origin", taken),
        forward = ex -> map_expr(x -> x isa Expr && x.head == :call && x.args[1] isa Symbol &&
                                      x.args[1] in rebased_def_names ?
                                   Expr(:call, x.args..., :p0) : x, ex),
        rebase = s -> forward(_anchor_relative_expr(s, gx, gy, :p0))
      push!(origin_stmts, Expr(:(=), origin, Expr(:call, :xy, gx, gy)))
      # Rebase the eligible global defs in place (they live in `out`).
      for (i, s) in enumerate(out)
        (s isa Expr && s.head == :function && s.args[1] isa Expr && s.args[1].head == :call &&
         s.args[1].args[1] in rebased_def_names) || continue
        let body = rebase(s.args[2])
          _assert_no_world_xy(body, s.args[1].args[1])
          out[i] = Expr(:function,
                        Expr(:call, s.args[1].args..., Expr(:kw, :p0, origin)),
                        body)
        end
      end
      for v in ordered
        push!(defs, Expr(:toplevel_comment,
                         "Storey $(fnames[v]) ($(v) = $(round(level_heights[v], digits=3)))"))
        let body = [rebase(s) for s in sections[v]]
          foreach(b -> _assert_no_world_xy(b, fnames[v]), body)
          push!(defs, Expr(:function,
                           Expr(:call, fnames[v], Expr(:kw, :p0, origin)),
                           Expr(:block, body...)))
        end
      end
    end
  end
  calls = Any[Expr(:toplevel_comment, "Build the storeys, bottom-up")]
  for v in ordered
    push!(calls, Expr(:call, fnames[v]))
  end
  # The origin parameter and the defs go where the first sectioned statement
  # stood; the CALLS go after every remaining global statement (sectioned
  # group_instances reference group vars defined in the global tail) but before
  # a trailing finalize_groups().
  tail_split = (!isempty(out) && out[end] isa Expr && out[end].head == :call &&
                out[end].args[1] == :finalize_groups) ? length(out) - 1 : length(out)
  Expr(:block, out[1:min(first_section_at - 1, tail_split)]..., origin_stmts..., defs...,
       out[min(first_section_at, tail_split + 1):tail_split]...,
       calls...,
       out[tail_split+1:end]...)
end

# 3.10 Parametrize the level SERIES — make n_levels a LIVE cardinality knob. parametrize_levels
# (3.7a) emits `n_levels = k`, but the level assigns stayed individually enumerated, so editing
# the flagship count parameter changed nothing. Rewrite
#   level_0 = level(0.0); level_1 = level(floor_height); …
# into the exemplar idiom (Isenberg's `for i in 1:floors`, Astana's division-driven levels):
#   levels = [level(i * floor_height) for i = 0:n_levels - 1]
#   level_0 = levels[1]; level_1 = levels[2]; …
# (with base_level_height when present), and when detect_level_repetition emitted the
# full-building typical-floor loop, retie `building_levels = levels` and the loop bound to
# `1:n_levels - 1` — so editing n_levels adds real levels AND replicates the typical storeys.
# Runs AFTER sectionalize_by_storey, whose _static_eval needs the scalar level(...) forms.
# Unconnected markers are untouched (already floor_height-relative or deliberately literal).
function parametrize_level_series(e::Expr)
  stmts = collect(stmts_of(e))
  nidx = findfirst(s -> s isa Expr && s.head == :(=) && s.args[1] == :n_levels &&
                        s.args[2] isa Int, stmts)
  nidx === nothing && return e
  n = stmts[nidx].args[2]
  li = [i for (i, s) in enumerate(stmts) if _is_level_assign(s)]
  length(li) == n || return e
  all(k -> stmts[li[k]].args[1] == Symbol("level_$(k-1)"), 1:n) || return e
  # Conservative collision guard: bail rather than mint a second `levels` binding.
  any(s -> s isa Expr && s.head == :(=) && s.args[1] == :levels, stmts) && return e
  let has_base = any(s -> s isa Expr && s.head == :(=) && s.args[1] == :base_level_height, stmts),
      hexpr = has_base ? :(base_level_height + i * floor_height) : :(i * floor_height),
      compr = Expr(:comprehension,
                   Expr(:generator, Expr(:call, :level, hexpr),
                        Expr(:(=), :i, Expr(:call, :(:), 0, Expr(:call, :-, :n_levels, 1))))),
      series_vect = [Symbol("level_$(k-1)") for k in 1:n],
      out = Any[]
    for (i, s) in enumerate(stmts)
      if i == li[1]
        push!(out, Expr(:(=), :levels, compr))
        for k in 1:n
          push!(out, Expr(:(=), series_vect[k], Expr(:ref, :levels, k)))
        end
      elseif i in li
        # consumed into the series above
      elseif s isa Expr && s.head == :(=) && s.args[1] == :building_levels &&
             s.args[2] isa Expr && s.args[2].head == :vect && s.args[2].args == series_vect
        push!(out, Expr(:(=), :building_levels, :levels))
      elseif s isa Expr && s.head == :for && s.args[1] isa Expr &&
             s.args[1].args[2] isa Expr && s.args[1].args[2].head == :call &&
             s.args[1].args[2].args[1] == :(:) && s.args[1].args[2].args[2] == 1 &&
             s.args[1].args[2].args[3] == n - 1 &&
             !isempty(collect_exprs(x -> x == :building_levels, s))
        push!(out, Expr(:for,
                        Expr(:(=), s.args[1].args[1],
                             Expr(:call, :(:), 1, Expr(:call, :-, :n_levels, 1))),
                        s.args[2]))
      else
        push!(out, s)
      end
    end
    Expr(:block, out...)
  end
end

# 3.11 Extract coordinate dimensions — after the anchor rebase, add_xy/add_xyz deltas are clean
# plan dimensions: the room width appears as the same 6.0 in every wall endpoint and the slab
# boundary. Mine recurring non-zero numeric x/y deltas (axis-scoped and signed: a 6.0 width is
# not a 6.0 depth), weighted by static loop lengths and def call counts like the kw mining, and
# lift each snap-bucket whose weight reaches the threshold into plan_width/plan_depth (+_k)
# parameters. This is the exemplar idiom (Isenberg's building_width/inner_radius, Astana's
# block_l/block_h feeding dozens of expressions): editing plan_width now resizes walls AND the
# slab together instead of requiring eight coordinated literal edits. z deltas belong to the
# levels and are never mined.
_collect_coord_dims(x, counts, mult, ctx) = nothing
function _collect_coord_dims(e::Expr, counts, mult, ctx)
  if e.head == :call && e.args[1] in (:add_xy, :add_xyz) && length(e.args) >= 4 && e.args[2] == :p0
    e.args[3] isa AbstractFloat && e.args[3] != 0.0 &&
      (counts[(:x, Float64(e.args[3]))] = get(counts, (:x, Float64(e.args[3])), 0) + mult)
    e.args[4] isa AbstractFloat && e.args[4] != 0.0 &&
      (counts[(:y, Float64(e.args[4]))] = get(counts, (:y, Float64(e.args[4])), 0) + mult)
  end
  let m = mult * _stmt_weight(e, ctx)
    for a in e.args
      _collect_coord_dims(a, counts, m, ctx)
    end
  end
  nothing
end

_replace_coord_dims(x, table) = x
_replace_coord_dims(e::Expr, table) =
  e.head == :call && e.args[1] in (:add_xy, :add_xyz) && length(e.args) >= 4 && e.args[2] == :p0 ?
    Expr(:call, e.args[1], :p0,
         e.args[3] isa AbstractFloat && haskey(table, (:x, Float64(e.args[3]))) ?
           table[(:x, Float64(e.args[3]))] : e.args[3],
         e.args[4] isa AbstractFloat && haskey(table, (:y, Float64(e.args[4]))) ?
           table[(:y, Float64(e.args[4]))] : e.args[4],
         e.args[5:end]...) :
    Expr(e.head, [_replace_coord_dims(a, table) for a in e.args]...)

# On a REAL floor plan almost every wall x-offset recurs ≥4 times (aligned walls), so an
# uncapped mining lifts dozens of position knobs — parameter spam, worse than the literals it
# replaces (live-corpus moradia3 emitted plan_width..plan_width_30). Only the DOMINANT extents
# are dimensions in the exemplar sense; the rest are positions and stay literal.
const _coord_dim_max_per_axis = 2

function extract_coordinate_dimensions(e::Expr)
  counts = Dict{Tuple{Symbol, Float64}, Int}()
  _collect_coord_dims(e, counts, 1, _dim_weight_context(e))
  used = Set{Symbol}(s.args[1] for s in stmts_of(e)
                     if s isa Expr && s.head == :(=) && s.args[1] isa Symbol)
  table = Dict{Tuple{Symbol, Float64}, Symbol}()
  assigns = Expr[]
  for (axis, base) in ((:x, "plan_width"), (:y, "plan_depth"))
    let pairs = [(v, c) for ((ax, v), c) in counts if ax == axis],
        kept = sort([b for b in _snap_buckets(pairs) if b.total >= _dim_min_occurrences],
                    by = b -> (-b.total, b.rep))
      for (i, b) in enumerate(kept[1:min(length(kept), _coord_dim_max_per_axis)])
        let name = _fresh_param(i == 1 ? base : "$(base)_$(i)", used)
          push!(assigns, Expr(:(=), name, b.rep))
          for v in b.members
            table[(axis, v)] = name
          end
        end
      end
    end
  end
  isempty(assigns) && return e
  Expr(:block, assigns..., stmts_of(_replace_coord_dims(e, table))...)
end

# 3.12 Derive parameter relations — exemplar parameter blocks define derived values (wall tops
# at -slab_thickness; Astana's column_d = 2*column_r). For each prologue Float parameter, try to
# express it via an EARLIER parameter as -p, k*p or p/k (k ∈ 2:4) within combined meta_program
# rounding (1e-6 relative), and rewrite the RHS symbolically, so editing the independent knob
# propagates. Guarded against numerology, with live-corpus evidence shaping the limits:
# - only _distinctive (non-round) values participate (6.0 = 2 * 3.0 is a coincidence);
# - identity relations are skipped (two same-valued dims stay independent knobs);
# - SUM/DIFFERENCE forms are deliberately absent: on real models they misfire (live moradia T4
#   derived floor_height = wall_base_offset - (wall_top_offset + wall_top_offset_2) — the
#   storey height coupled to wall offsets). The exemplar outer = inner + width relation is
#   semantically identifiable only by a human;
# - STRUCTURAL params (floor_height, base_level_height, *_spacing — owned by
#   parametrize_levels/symbolize_ranges) are never TARGETS, though they remain sources
#   (parapet datums already derive from floor_height upstream);
# - mined plan_* params are POSITIONS (aligned-wall offsets), not lengths — excluded as
#   targets AND sources (live moradia3 derived a depth from two widths).
function derive_parameter_relations(e::Expr)
  stmts = collect(stmts_of(e))
  mined = sym -> startswith(String(sym), "plan_")
  structural = sym -> sym in (:floor_height, :base_level_height) ||
                      endswith(String(sym), "_spacing")
  isparam = s -> s isa Expr && s.head == :(=) && s.args[1] isa Symbol &&
                 s.args[2] isa AbstractFloat && !mined(s.args[1])
  rel_tol(v) = max(1e-9, 1e-6 * abs(v))
  out = collect(stmts)
  earlier = Tuple{Symbol, Float64}[]
  changed = false
  for (i, s) in enumerate(stmts)
    # Only the LEADING parameter block participates: params defined mid-program
    # next to protected constructs (frame_width beside its frame_family) must
    # not get numerological couplings to unrelated knobs.
    (s isa Expr && s.head == :(=) && s.args[1] isa Symbol && s.args[2] isa Real) || break
    isparam(s) || continue
    let v = Float64(s.args[2])
      if _distinctive(v) && !structural(s.args[1])
        found = nothing
        for (p, pv) in earlier
          _distinctive(pv) || continue
          abs(v - pv) <= rel_tol(v) && continue                    # identity: keep knobs separate
          if abs(v + pv) <= rel_tol(v)
            found = Expr(:call, :-, p)
          else
            for k in 2:4
              abs(v - k * pv) <= rel_tol(v) && (found = Expr(:call, :*, k, p))
              abs(v - pv / k) <= rel_tol(v) && (found = Expr(:call, :/, p, k))
              found === nothing || break
            end
          end
          found === nothing || break
        end
        if found !== nothing
          out[i] = Expr(:(=), s.args[1], found)
          changed = true
        end
      end
      push!(earlier, (s.args[1], v))
    end
  end
  changed ? Expr(:block, out...) : e
end

# 3.13 Round parameters to architectural values — introspection yields float noise no human
# would write (floor_height = 3.2599999, wall_top_offset = -0.19999999; a designer writes 3.26
# and -0.2). Snap each top-level Float parameter to the ROUNDEST decimal within
# _param_round_tolerance (0.1 mm — noise-sized, far inside the passes' 1e-3 snap class and the
# oracle's 2e-3 tolerance) and keep the introspected value in a trailing comment:
#   floor_height = 3.26  # rounded from the introspected 3.2599999
# Every use bound to the parameter shifts coherently with it; values with no round neighbor
# (a mined 0.241434 position) stay verbatim, and already-round values get no comment. The
# comment rides on an :inline_comment pseudo-node (printer + wrap + the oracle's evaluator
# understand it; parsing the printed source strips it naturally).
const _param_round_tolerance = 1e-4

_architectural_round(v::Float64) =
  let hit = nothing
    for d in 0:3
      let r = round(v, digits = d)
        if abs(r - v) <= _param_round_tolerance
          hit = r
          break
        end
      end
    end
    hit
  end

function round_parameters(e::Expr)
  changed = false
  out = map(collect(stmts_of(e))) do s
    (s isa Expr && s.head == :(=) && s.args[1] isa Symbol && s.args[2] isa AbstractFloat) ||
      return s
    let v = Float64(s.args[2]), r = _architectural_round(v)
      if r === nothing || r == v
        s
      else
        changed = true
        Expr(:inline_comment,
             Expr(:(=), s.args[1], r),
             "rounded from the introspected $(s.args[2])")
      end
    end
  end
  changed ? Expr(:block, out...) : e
end

# ─── The standard pass list ──────────────────────────────────────────────────
# One authoritative order shared by the BIM (Revit) and geometric (AutoCAD) pipelines so they
# cannot drift. `header` is the backend-appropriate header pass; `wrap` inserts the optional
# function wrapper before it.
codegen_passes(b, fmap; header=add_header(b), wrap=false,
               level_names=Dict{Float64, String}()) =
  let passes = Any[extract_levels,
                   extract_families(fmap),
                   add_backend_families(b, fmap),
                   hoist_opening_frames,
                   detect_level_repetition,
                   extract_functions,
                   sort_statements,
                   loop_rerolling,
                   parametrize_levels,
                   symbolize_ranges,
                   symbolize_angles,
                   extract_shared_dimensions,
                   unify_constants,
                   sectionalize_by_storey(level_names),
                   parametrize_level_series,
                   extract_coordinate_dimensions,
                   derive_parameter_relations,
                   round_parameters]
    wrap && push!(passes, wrap_program_in_function)
    push!(passes, header)
    passes
  end

# 3.6 Add header
# The timestamp line is gated: it makes otherwise-identical generations differ, which breaks
# golden-based comparison (the stress harness and tests run with it off).
const codegen_emit_timestamp = Parameter(true)

_header_stmts(title, b; obj_resources=false) =
  let stmts = Any[Expr(:toplevel_comment, title)]
    codegen_emit_timestamp() &&
      push!(stmts, Expr(:toplevel_comment, "Generated on: $(Dates.now())"))
    push!(stmts, Expr(:using, Expr(:., b_codegen_module(b))))
    # When the program carries extracted OBJ/.rfa content, register its sibling folder so relative
    # obj_family/obj_model names resolve wherever the generated file lives.
    obj_resources &&
      push!(stmts, Expr(:call, :add_resource_folder!,
                        Expr(:call, :joinpath,
                             Expr(:macrocall, Symbol("@__DIR__"), nothing),
                             "khepri_obj_models")))
    stmts
  end

# The exemplar ADs open with a labeled parameter section (Isenberg/Astana's
# "# PARAMETERS" headers); when the program leads with knobs, name the block.
_with_parameter_comment(stmts) =
  !isempty(stmts) && stmts[1] isa Expr && stmts[1].head == :(=) &&
  stmts[1].args[1] isa Symbol && stmts[1].args[2] isa Real ?
    Any[Expr(:toplevel_comment, "Parameters"), stmts...] : stmts

add_header(b; obj_resources=false) = (e::Expr) ->
  Expr(:block,
       _header_stmts("Auto-generated Khepri code from Revit model", b; obj_resources=obj_resources)...,
       _with_parameter_comment(stmts_of(e))...)

# Header for the geometric round-trip (own provenance line so the BIM add_header — and its goldens —
# stay untouched).
add_geometric_header(b) = (e::Expr) ->
  Expr(:block,
       _header_stmts("Auto-generated Khepri code (geometric round-trip via $(b_codegen_module(b)))", b)...,
       _with_parameter_comment(stmts_of(e))...)

stmts_of(e::Expr) = e.head == :block ? e.args : [e]

# ─── Running a generated program on a backend ────────────────────────────────
# Statement-by-statement execution of a generated AD program: binds the backend's guard symbol
# (so `@isdefined(threejs) && …` mappings fire), selects the backend, skips the `using` line
# (the caller's session already has its backend package loaded), and contains per-statement
# failures so one bad element skips instead of aborting the run. Returns counts + deduped errors.
function run_generated_program(path::AbstractString; b=top_backend(), skip_using=true)
  # Bind BOTH the backend's display name and its lowercase form: the generated guards use the
  # lowercase package-idiomatic symbol (`@isdefined(threejs)`), while backend_name(b) may be
  # capitalized ("Threejs") — binding only the latter silently disabled every family mapping.
  for name in unique([Symbol(backend_name(b)), Symbol(lowercase(backend_name(b)))])
    isdefined(Main, name) || Core.eval(Main, Expr(:(=), name, b))
  end
  current_backend(b)
  let ok = 0, failed = 0,
      errors = Dict{String, Int}(),
      # Statement-wise eval has no source file, so `@__DIR__` in the program (its
      # khepri_obj_models references) resolves to pwd — anchor pwd at the program's home.
      prev_dir = pwd()
    cd(dirname(abspath(path)))
    try
      for e in Meta.parseall(read(basename(path), String)).args
        e isa LineNumberNode && continue
        skip_using && e isa Expr && e.head == :using && continue
        try
          Core.eval(Main, e)
          ok += 1
        catch ex
          failed += 1
          let msg = first(sprint(showerror, ex), 140)
            errors[msg] = get(errors, msg, 0) + 1
          end
        end
      end
    finally
      cd(prev_dir)
    end
    (ok=ok, failed=failed, errors=sort!(collect(errors), by=last, rev=true))
  end
end

# ─── Phase 4: Pretty-Printing ─────────────────────────────────────────────────

function expr_to_string(e::Expr; indent=0)
  let ind = "  " ^ indent,
      io = IOBuffer()
    _print_block(io, e, indent)
    String(take!(io))
  end
end

function _print_block(io, e::Expr, indent)
  let stmts = e.head == :block ? e.args : [e],
      prev_section = :none
    for (i, s) in enumerate(stmts)
      let cur_section = _stmt_section(s)
        # Add blank line between sections
        if i > 1 && cur_section != prev_section && prev_section != :none
          println(io)
        end
        _print_stmt(io, s, indent)
        println(io)
        prev_section = cur_section
      end
    end
  end
end

_stmt_section(s) =
  if s isa Expr
    if s.head == :inline_comment
      _stmt_section(s.args[1])
    elseif s.head == :toplevel_comment
      :header
    elseif s.head == :using
      :header
    elseif s.head == :call && s.args[1] == :add_resource_folder!
      :header
    elseif s.head == :(=) && s.args[2] isa Real
      :params
    elseif s.head == :(=) && s.args[2] isa Expr && s.args[2].head == :vect &&
           all(a -> a isa Symbol, s.args[2].args)
      :levels
    elseif s.head == :(=) && s.args[2] isa Expr && s.args[2].head == :comprehension
      # The parametrize_level_series comprehension IS the level block.
      :levels
    elseif s.head == :(=) && s.args[2] isa Expr && s.args[2].head == :ref &&
           s.args[2].args[1] == :levels
      :levels
    elseif s.head == :(=) && s.args[2] === :levels
      :levels
    elseif s.head == :(=) && s.args[2] isa Expr
      let rhs = s.args[2]
        if rhs.head == :call && rhs.args[1] in (:level, :unconnected_level)
          :levels
        elseif rhs.head == :call && rhs.args[1] in _family_function_names
          :families
        elseif rhs.head == :call && rhs.args[1] in (:xy, :xyz)
          # A coordinate-valued parameter (building_origin) belongs with the knobs.
          :params
        else
          :elements
        end
      end
    elseif s.head == :call && s.args[1] == :set_backend_family
      :families
    elseif s.head == :&& && s.args[2] isa Expr && s.args[2].head == :call &&
           s.args[2].args[1] == :set_backend_family
      :families
    elseif s.head == :function
      :groups
    elseif s.head == :(=) && s.args[2] isa Expr && s.args[2].head == :call &&
           s.args[2].args[1] == :group
      :groups
    elseif s.head == :call && s.args[1] == :group_instance
      :groups
    elseif s.head == :for
      :elements
    elseif s.head == :call
      :elements
    else
      :other
    end
  elseif s isa Symbol
    # A bare trailing symbol (the `__wall` a let-block returns) belongs with the
    # elements around it — no blank separator line before it.
    :elements
  else
    :other
  end

# Check if an Expr contains a :do block (for pretty-printing with do...end)
_has_do_block(e::Expr) = e.head == :call && any(a -> a isa Expr && a.head == :do, e.args) ||
                          e.head == :do ||
                          (e.head == :call && any(a -> a isa Expr && _has_do_block(a), e.args))
_has_do_block(x) = false

# Print a call expression that contains a do block:
# fn(args..., inner_fn() do\n  body\nend)
function _print_do_call(io, e::Expr, indent)
  let ind = "  " ^ indent
    if e.head == :call
      # Find do-block argument
      let do_idx = findfirst(a -> a isa Expr && a.head == :do, e.args),
          fn = e.args[1],
          regular_args = [a for (i, a) in enumerate(e.args[2:end]) if i + 1 != do_idx]
        if do_idx !== nothing
          let do_expr = e.args[do_idx],
              do_call = do_expr.args[1],  # the inner function call (e.g., collecting_shapes)
              do_lambda = do_expr.args[2], # the -> with body
              do_body = do_lambda.args[2]  # the block body
            if isempty(regular_args)
              print(io, "$(_expr_str(fn))($(_expr_str(do_call))) do")
            else
              print(io, "$(_expr_str(fn))($(join(map(_expr_str, regular_args), ", ")), $(_expr_str(do_call))) do")
            end
            println(io)
            _print_block(io, do_body, indent + 1)
            print(io, "$(ind)end")
          end
        else
          print(io, _expr_str(e))
        end
      end
    else
      print(io, _expr_str(e))
    end
  end
end

function _print_stmt(io, s::Expr, indent)
  let ind = "  " ^ indent
    if s.head == :toplevel_comment
      print(io, "$(ind)# $(s.args[1])")
    elseif s.head == :inline_comment
      # A statement with a trailing provenance comment (round_parameters).
      _print_stmt(io, s.args[1], indent)
      print(io, "  # $(s.args[2])")
    elseif s.head == :using
      print(io, "$(ind)using $(join(map(_expr_str, s.args), ", "))")
    elseif s.head == :(=)
      let rhs = s.args[2]
        if _has_do_block(rhs)
          print(io, "$(ind)$(s.args[1]) = ")
          _print_do_call(io, rhs, indent)
        else
          print(io, "$(ind)$(s.args[1]) = $(_expr_str(rhs))")
        end
      end
    elseif s.head == :function
      let sig = s.args[1],
          body = s.args[2]
        print(io, "$(ind)function $(_sig_str(sig))")
        println(io)
        _print_block(io, body, indent + 1)
        print(io, "$(ind)end")
      end
    elseif s.head == :for
      let binding = s.args[1],
          body = s.args[2]
        print(io, "$(ind)for $(_expr_str(binding))")
        println(io)
        _print_block(io, body, indent + 1)
        print(io, "$(ind)end")
      end
    elseif s.head == :let
      # Without this arm, let-blocks fall through to Base's string(e): 4-space
      # body and a column-0 `end` that visually terminates the enclosing storey
      # function early.
      let binding = s.args[1],
          body = s.args[2]
        print(io, "$(ind)let $(_expr_str(binding))")
        println(io)
        _print_block(io, body, indent + 1)
        print(io, "$(ind)end")
      end
    elseif s.head == :call
      _print_call(io, s, ind)
    else
      print(io, "$(ind)$(_expr_str(s))")
    end
  end
end

_print_stmt(io, s, indent) = print(io, "  " ^ indent, _expr_str(s))

# Function signature: keyword parameters print after `;` (a :call's generic printing would render
# them as optional positionals).
_sig_str(sig) =
  let fn = sig.args[1],
      args = sig.args[2:end],
      pb = findfirst(a -> a isa Expr && a.head == :parameters, args),
      pos = pb === nothing ? args : [args[1:pb-1]..., args[pb+1:end]...],
      kw = pb === nothing ? [] : args[pb].args
    isempty(kw) ?
      "$(fn)($(join(map(_expr_str, pos), ", ")))" :
      "$(fn)($(join(map(_expr_str, pos), ", ")); $(join(map(_kw_str, kw), ", ")))"
  end

function _print_call(io, e::Expr, ind)
  let fn = e.args[1],
      args = e.args[2:end],
      # Extract keyword args from :parameters block if present
      params_block = findfirst(a -> a isa Expr && a.head == :parameters, args),
      extra_kw = params_block !== nothing ? args[params_block].args : Expr[],
      rest = params_block !== nothing ? [args[1:params_block-1]..., args[params_block+1:end]...] : args,
      pos_args = filter(a -> !(a isa Expr && a.head == :kw), rest),
      kw_args = vcat(filter(a -> a isa Expr && a.head == :kw, rest), extra_kw),
      short = _expr_str(e)
    if length(short) + length(ind) <= 80
      print(io, "$(ind)$(short)")
    else
      # Multi-line format
      print(io, "$(ind)$(fn)(")
      let parts = vcat(map(_expr_str, pos_args), map(_kw_str, kw_args)),
          first_line = "$(fn)("
        for (i, p) in enumerate(parts)
          if i == 1
            print(io, p)
          else
            print(io, ",\n$(ind)  $(p)")
          end
        end
        print(io, ")")
      end
    end
  end
end

_kw_str(e::Expr) = "$(e.args[1])=$(_expr_str(e.args[2]))"

# Infix arithmetic printing (parametrized code: `base_level_height + 2 * floor_height`).
_arith_prec(op) = op in (:+, :-) ? 1 : 2

_arith_arg_str(op, a, first_arg) =
  let s = _expr_str(a)
    a isa Expr && a.head == :call && a.args[1] in (:+, :-, :*, :/) &&
    (_arith_prec(a.args[1]) < _arith_prec(op) || (!first_arg && op in (:-, :/))) ?
      "($s)" : s
  end

function _expr_str(e::Expr)
  if e.head == :call && e.args[1] === Symbol(":") && length(e.args) in (3, 4)
    # Range operator is infix: (:)(a,b) → a:b, (:)(a,step,b) → a:step:b (a for-loop reroll range).
    join(map(_expr_str, e.args[2:end]), ":")
  elseif e.head == :call && e.args[1] in (:+, :-, :*, :/) && length(e.args) >= 2
    length(e.args) == 2 ?
      (e.args[1] == :- ?
         # -(a/b) == (-a)/b and -(a*b) == (-a)*b exactly, so a negated product or
         # quotient prints without the parens: "-pi / 2", not "-(pi / 2)".
         (e.args[2] isa Expr && e.args[2].head == :call && e.args[2].args[1] in (:*, :/) ?
            "-$(_expr_str(e.args[2]))" :
            "-$(_arith_arg_str(:-, e.args[2], false))") :
         "$(e.args[1])($(_expr_str(e.args[2])))") :
      join([_arith_arg_str(e.args[1], a, i == 1) for (i, a) in enumerate(e.args[2:end])],
           " $(e.args[1]) ")
  elseif e.head == :call
    let fn = e.args[1],
        args = e.args[2:end],
        # Extract keyword args from :parameters block if present
        params_block = findfirst(a -> a isa Expr && a.head == :parameters, args),
        extra_kw = params_block !== nothing ? args[params_block].args : Expr[],
        rest = params_block !== nothing ? [args[1:params_block-1]..., args[params_block+1:end]...] : args,
        pos = filter(a -> !(a isa Expr && a.head == :kw), rest),
        kw = vcat(filter(a -> a isa Expr && a.head == :kw, rest), extra_kw),
        parts = vcat(map(_expr_str, pos), map(_kw_str, kw))
      "$(fn)($(join(parts, ", ")))"
    end
  elseif e.head == :(=)
    "$(_expr_str(e.args[1])) = $(_expr_str(e.args[2]))"
  elseif e.head == :kw
    "$(e.args[1])=$(_expr_str(e.args[2]))"
  elseif e.head == :vect
    "[$(join(map(_expr_str, e.args), ", "))]"
  elseif e.head == :tuple
    "($(join(map(_expr_str, e.args), ", ")))"
  elseif e.head == :ref
    "$(_expr_str(e.args[1]))[$(join(map(_expr_str, e.args[2:end]), ", "))]"
  elseif e.head == :.
    join(map(a -> a isa QuoteNode ? string(a.value) : _expr_str(a), e.args), ".")
  elseif e.head == :macrocall
    let macro_name = string(e.args[1])
      if macro_name == "@raw_str"
        "raw\"$(e.args[end])\""
      elseif macro_name == "@isdefined"
        # Parenthesize so `@isdefined(revit) && …` doesn't greedily consume the `&&` as the macro arg.
        "@isdefined($(_expr_str(e.args[end])))"
      elseif macro_name == "@__DIR__"
        "@__DIR__"
      else
        string(e)
      end
    end
  elseif e.head == :&&
    "$(_expr_str(e.args[1])) && $(_expr_str(e.args[2]))"
  elseif e.head == :using
    "using $(join(map(_expr_str, e.args), ", "))"
  elseif e.head == :do
    # do block: args[1] is the call, args[2] is the -> lambda
    let call_str = _expr_str(e.args[1]),
        body = e.args[2].args[2],
        body_stmts = body.head == :block ? body.args : [body]
      "$(call_str) do; $(join(map(_expr_str, body_stmts), "; ")); end"
    end
  elseif e.head == :block
    join(map(_expr_str, e.args), "; ")
  else
    string(e)
  end
end

_expr_str(s::Symbol) = string(s)
_expr_str(x::Real) = string(meta_program(x))
_expr_str(x::Int) = string(x)
_expr_str(x::AbstractString) = repr(x)
_expr_str(x::Bool) = string(x)
_expr_str(x::QuoteNode) = string(x.value)
_expr_str(::Nothing) = "nothing"
_expr_str(x) = string(x)

# ─── Pass-equivalence oracle ─────────────────────────────────────────────────
#=
The transformation passes must PRESERVE GEOMETRY: the raw `model_to_expr` output
and the transformed program, both realized on a MeasureBackend, must produce the
same shape multiset (within the passes' declared snapping tolerance — the reroll
passes snap values to arithmetic progressions within ~1e-3). The oracle folds the
pipeline pass-by-pass, so a regression names the offending pass directly. Fully
headless: no Revit, no golden files; this is the layer that catches the
translate/reorder/reroll class of bugs (a grouped slab landing at double its
offset, a rerolled loop dropping an element) in CI.

Statement-level realization errors in ANY stage are failures — the raw program
is the semantic reference and must realize cleanly.
=#
export pass_equivalence_report, codegen_pass_names

codegen_pass_names(; wrap=false) =
  let names = ["extract_levels", "extract_families", "add_backend_families", "hoist_opening_frames",
               "detect_level_repetition", "extract_functions", "sort_statements",
               "loop_rerolling", "parametrize_levels", "symbolize_ranges",
               "symbolize_angles", "extract_shared_dimensions", "unify_constants",
               "sectionalize_by_storey", "parametrize_level_series",
               "extract_coordinate_dimensions", "derive_parameter_relations",
               "round_parameters"]
    wrap && push!(names, "wrap_program_in_function")
    push!(names, "header")
    names
  end

function pass_equivalence_report(b, model; tol_pos=2e-3, tol_size=2e-3,
                                 passes=nothing, pass_names=nothing)
  let fmap = family_expr_map(model),
      raw = model_to_expr(model),
      # The header pass adds `using`/comments only — geometry-neutral by
      # construction, but included anyway: the oracle re-verifies it.
      passes = passes === nothing ? codegen_passes(b, fmap) : passes,
      pass_names = pass_names === nothing ? codegen_pass_names() : pass_names,
      # Per-SHAPE granularity: a rerolled for-loop or an extracted function call
      # registers one shape per created element, exactly like the unrolled raw
      # program — per-statement aggregation would spuriously mismatch after
      # loop_rerolling collapses N statements into one.
      stage_measurements = e ->
        let mb = measure_backend(),
            rows = measured_statements(e; b=mb),
            errs = [(r.index, r.error) for r in rows if r.error !== nothing]
          ([r.measurement for r in measured_shapes(mb) if r.measurement.n_trigs > 0],
           errs)
        end,
      (base, base_errs) = stage_measurements(raw)
    isempty(base_errs) ||
      return (ok=false, failing_pass=0, pass_name="raw (model_to_expr)",
              errors=base_errs, unmatched_raw=Int[], unmatched_stage=Int[],
              n_raw=length(base), n_stage=0)
    let stage = raw
      for (k, p) in enumerate(passes)
        stage = p(stage)
        let (ms, errs) = stage_measurements(stage),
            res = match_measurements(base, ms; tol_pos=tol_pos, tol_size=tol_size)
          if !isempty(errs) || !isempty(res.unmatched_a) || !isempty(res.unmatched_b)
            return (ok=false, failing_pass=k,
                    pass_name=k <= length(pass_names) ? pass_names[k] : "pass $k",
                    errors=errs,
                    unmatched_raw=res.unmatched_a, unmatched_stage=res.unmatched_b,
                    n_raw=length(base), n_stage=length(ms))
          end
        end
      end
      (ok=true, failing_pass=nothing, pass_name=nothing, errors=[],
       unmatched_raw=Int[], unmatched_stage=Int[],
       n_raw=length(base), n_stage=length(base))
    end
  end
end
