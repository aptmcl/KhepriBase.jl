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
  if (e.head == :kw && e.args[1] in (:doors, :windows, :family)) ||
     (e.head == :call && (e.args[1] in (:add_door, :add_window) || e.args[1] in _family_function_names))
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
        body = Expr(:block, [_symbolize_for(b, params, used) for b in stmts_of(s.args[2])]...)
      if rng isa Expr && rng.head == :call && rng.args[1] == :range
        let step_kw = findfirst(a -> a isa Expr && a.head == :kw && a.args[1] == :step, rng.args),
            len_kw = findfirst(a -> a isa Expr && a.head == :kw && a.args[1] == :length, rng.args)
          if step_kw !== nothing && len_kw !== nothing
            let callee = _loop_body_callee(body),
                axis = _loop_var_axis(body, var),
                base = callee === nothing ? "items" :
                       let cs = string(callee)
                         endswith(cs, "s") ? cs : cs * "s"
                       end,
                suffix = axis === nothing ? "" : "_$(axis)",
                nsym = _fresh_param("n_$(base)$(suffix)", used),
                ssym = _fresh_param("$(base)$(suffix)_spacing", used)
              push!(params, Expr(:(=), ssym, rng.args[step_kw].args[2]))
              push!(params, Expr(:(=), nsym, rng.args[len_kw].args[2]))
              Expr(:for,
                   Expr(:(=), var,
                        Expr(:call, :range, rng.args[2],
                             Expr(:kw, :step, ssym), Expr(:kw, :length, nsym))),
                   body)
            end
          else
            Expr(:for, binding, body)
          end
        end
      else
        Expr(:for, binding, body)
      end
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
  (e.head == :call &&
   (e.args[1] in _family_function_names || e.args[1] in (:add_door, :add_window, :level,
                                                         :set_backend_family, :obj_model))) ?
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

# 3.8 Optional: wrap the whole program in `function building(; params...) … end; building()` so the
# parameters become re-invocable keyword arguments. Off by default (generate_khepri_code kwarg).
function wrap_program_in_function(e::Expr; name=:building)
  stmts = collect(stmts_of(e))
  isparam = s -> s isa Expr && s.head == :(=) && s.args[1] isa Symbol && s.args[2] isa Real
  k = something(findfirst(s -> !isparam(s), stmts), length(stmts) + 1) - 1
  k == 0 && return e
  Expr(:block,
       Expr(:function,
            Expr(:call, name,
                 Expr(:parameters, [Expr(:kw, s.args[1], s.args[2]) for s in stmts[1:k]]...)),
            Expr(:block, stmts[k+1:end]...)),
       Expr(:call, name))
end

# ─── The standard pass list ──────────────────────────────────────────────────
# One authoritative order shared by the BIM (Revit) and geometric (AutoCAD) pipelines so they
# cannot drift. `header` is the backend-appropriate header pass; `wrap` inserts the optional
# function wrapper before it.
codegen_passes(b, fmap; header=add_header(b), wrap=false) =
  let passes = Any[extract_levels,
                   extract_families(fmap),
                   add_backend_families(b, fmap),
                   detect_level_repetition,
                   extract_functions,
                   sort_statements,
                   loop_rerolling,
                   parametrize_levels,
                   symbolize_ranges,
                   unify_constants]
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

add_header(b; obj_resources=false) = (e::Expr) ->
  Expr(:block,
       _header_stmts("Auto-generated Khepri code from Revit model", b; obj_resources=obj_resources)...,
       stmts_of(e)...)

# Header for the geometric round-trip (own provenance line so the BIM add_header — and its goldens —
# stay untouched).
add_geometric_header(b) = (e::Expr) ->
  Expr(:block,
       _header_stmts("Auto-generated Khepri code (geometric round-trip via $(b_codegen_module(b)))", b)...,
       stmts_of(e)...)

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
    if s.head == :toplevel_comment
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
    elseif s.head == :(=) && s.args[2] isa Expr
      let rhs = s.args[2]
        if rhs.head == :call && rhs.args[1] in (:level, :unconnected_level)
          :levels
        elseif rhs.head == :call && rhs.args[1] in _family_function_names
          :families
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
      (e.args[1] == :- ? "-$(_arith_arg_str(:-, e.args[2], false))" :
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
  let names = ["extract_levels", "extract_families", "add_backend_families",
               "detect_level_repetition", "extract_functions", "sort_statements",
               "loop_rerolling", "parametrize_levels", "symbolize_ranges",
               "unify_constants"]
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
