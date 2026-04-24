#####################################################################
# Constraints — typed, algebra-rich validation of architectural designs
#
# A `Constraint` wraps a check function with a severity (HARD/SOFT/
# PREFERENCE), a category, and a human-readable name. `validate` runs
# a vector of constraints against any context type and returns a
# `ValidationResult` grouping violations by severity. Constraints
# compose via `combine`, `either`, `when`, `with_severity`.
#
# This module is context-agnostic: a `Constraint.check(ctx)` can run
# against whatever `ctx` its constructor was built for — a KhepriBase
# `BuildResult`, an AlgorithmicArchitecture `LayoutResult`, or any
# user-defined structure. `validate(ctx, constraints)` dispatches to
# each check with that `ctx`.
#
# A `ConstraintFixer` lets `apply_fixers` iterate a fix loop: validate,
# find hard violations, apply the first matching fixer's transform to
# the description, and repeat until either the design is hard-clean,
# no fixer matches, or `max_iters` is reached.

export Constraint, ConstraintSet, Violation, ValidationResult,
       ConstraintSeverity, HARD, SOFT, PREFERENCE,
       ConstraintCategory, DIMENSIONAL, ADJACENCY, AREA_PROPORTION,
       CIRCULATION, ENVIRONMENTAL,
       validate, report,
       combine, either, when, with_severity, merge_constraints,
       ConstraintFixer, apply_fixers
# The constraint-*library* constructors (min_area, must_adjoin,
# all_reachable, facade_ratio, …) live in `ConstraintLibrary.jl`,
# which is loaded after `Spaces.jl` so they can dispatch on both
# `Layout` (declarative engine output) and `BuildResult` (imperative
# `build(storey)` output) contexts.

#=== Severity and Category ===#

"""
    ConstraintSeverity

Severity level for a constraint: `HARD` (must be satisfied), `SOFT`
(should be satisfied), or `PREFERENCE` (nice to have).
"""
@enum ConstraintSeverity HARD SOFT PREFERENCE

@doc "Hard-severity level: a violation blocks design approval." HARD
@doc "Soft-severity level: a violation is reported but does not block." SOFT
@doc "Preference-severity level: a nice-to-have, lowest weight in scoring." PREFERENCE

"""
    ConstraintCategory

Classification of what aspect a constraint checks: `DIMENSIONAL`,
`ADJACENCY`, `AREA_PROPORTION`, `CIRCULATION`, or `ENVIRONMENTAL`.
"""
@enum ConstraintCategory begin
  DIMENSIONAL
  ADJACENCY
  AREA_PROPORTION
  CIRCULATION
  ENVIRONMENTAL
end

@doc "Dimensional category: constraints on lengths, widths, heights, areas." DIMENSIONAL
@doc "Adjacency category: constraints on which kinds may or must neighbour." ADJACENCY
@doc "Area-proportion category: constraints on minimum or ratio areas." AREA_PROPORTION
@doc "Circulation category: constraints on reachability and egress paths." CIRCULATION
@doc "Environmental category: constraints on daylight, ventilation, exposure." ENVIRONMENTAL

#=== Core Types ===#

"""
    Violation

A single constraint violation recording the constraint name, severity,
category, the offending target, a message, and the actual vs. limit values.
"""
struct Violation
  constraint_name::String
  severity::ConstraintSeverity
  category::ConstraintCategory
  target::String
  message::String
  actual_value::Float64
  limit_value::Float64
end

"""
    Constraint

A named check paired with severity and category. The `check` function
takes a context (a `BuildResult`, a `LayoutResult`, or any user value)
and returns `Vector{Violation}`.
"""
struct Constraint
  name::String
  severity::ConstraintSeverity
  category::ConstraintCategory
  check::Function
end

"""
    ConstraintSet

A named collection of `Constraint`s.
"""
struct ConstraintSet
  name::String
  constraints::Vector{Constraint}
end

"""
    ValidationResult

Outcome of `validate`: whether the design passed (no hard violations),
lists of hard/soft/preference violations, and a weighted penalty score.
"""
struct ValidationResult
  passed::Bool
  hard_violations::Vector{Violation}
  soft_violations::Vector{Violation}
  preferences::Vector{Violation}
  score::Float64
end

#=== Validation ===#

"""
    validate(ctx, constraints)
    validate(ctx, cs::ConstraintSet)

Run every constraint against `ctx` and return a `ValidationResult`
grouping violations by severity. `ctx` can be any value; the individual
constraints' check functions decide how to query it.
"""
function validate(ctx, constraints::Vector{Constraint})
  hard = Violation[]
  soft = Violation[]
  prefs = Violation[]
  for c in constraints, v in c.check(ctx)
    if v.severity == HARD
      push!(hard, v)
    elseif v.severity == SOFT
      push!(soft, v)
    else
      push!(prefs, v)
    end
  end
  score = 1000.0 * length(hard) + 10.0 * length(soft) + 1.0 * length(prefs)
  ValidationResult(isempty(hard), hard, soft, prefs, score)
end

validate(ctx, cs::ConstraintSet) = validate(ctx, cs.constraints)

"""
    report(result; io=stdout)

Print a human-readable summary of a `ValidationResult`.
"""
function report(result::ValidationResult; io=stdout)
  println(io, "Status: ", result.passed ? "PASSED" : "FAILED",
          " (score: ", result.score, ")")
  for (label, vs) in [("HARD", result.hard_violations),
                       ("SOFT", result.soft_violations),
                       ("PREF", result.preferences)]
    isempty(vs) && continue
    println(io, "\n--- $label ($(length(vs))) ---")
    for v in vs
      println(io, "  [$(v.category)] $(v.target): $(v.message)")
    end
  end
end

#=== Algebra ===#

"""
    combine(constraints...)

Conjoin multiple constraints into one whose check returns the
concatenation of all inputs' violations. Takes the maximum severity.
"""
combine(constraints::Constraint...) = Constraint(
  join([c.name for c in constraints], " & "),
  maximum(c.severity for c in constraints),
  first(constraints).category,
  ctx -> vcat((c.check(ctx) for c in constraints)...))

"""
    either(a, b)

Disjoin two constraints: passes when at least one has no violations.
If both fail, the fewer violations are reported.
"""
either(a::Constraint, b::Constraint) = Constraint(
  "$(a.name) | $(b.name)",
  max(a.severity, b.severity),
  a.category,
  ctx -> let va = a.check(ctx), vb = b.check(ctx)
    isempty(va) || isempty(vb) ? Violation[] :
      length(va) <= length(vb) ? va : vb
  end)

"""
    when(predicate, constraint)

Gate a constraint so its check runs only when `predicate(ctx)` is true.
"""
when(predicate, constraint::Constraint) = Constraint(
  "when_$(constraint.name)",
  constraint.severity,
  constraint.category,
  ctx -> predicate(ctx) ? constraint.check(ctx) : Violation[])

"""
    with_severity(constraint, severity)

Return a copy of `constraint` with its severity (and every emitted
violation's severity) overridden to the given level.
"""
with_severity(constraint::Constraint, severity::ConstraintSeverity) = Constraint(
  constraint.name, severity, constraint.category,
  ctx -> [Violation(v.constraint_name, severity, v.category, v.target, v.message,
                    v.actual_value, v.limit_value)
          for v in constraint.check(ctx)])

"""
    merge_constraints(sets...)

Concatenate multiple `ConstraintSet`s into a single set.
"""
merge_constraints(sets::ConstraintSet...) = ConstraintSet(
  join([s.name for s in sets], " + "),
  vcat((s.constraints for s in sets)...))

#=== Fixer Loop ===#

"""
    ConstraintFixer(name, pattern, transform)

Attempt to repair a design in response to a constraint violation.
`pattern` is a substring matched against `Violation.constraint_name`;
on a match `transform(desc, violation)` is applied, returning a
rewritten description. See `apply_fixers` for the outer loop.
"""
struct ConstraintFixer
  name::String
  pattern::String
  transform::Function
end

"""
    apply_fixers(desc, build, constraints, fixers; max_iters=10)

Iteratively fix hard violations. Each iteration:
1. Build a context via `build(desc)`.
2. Validate against `constraints`.
3. For each hard violation, apply the first matching fixer.
4. Stop when hard-clean, no fixer matches, or `max_iters` is reached.

Returns the (possibly rewritten) `desc` together with the final
`ValidationResult`.
"""
function apply_fixers(desc, build, constraints, fixers;
                      max_iters::Integer = 10)
  local vr
  for _ in 1:max_iters
    ctx = build(desc)
    vr = validate(ctx, constraints)
    isempty(vr.hard_violations) && return (desc, vr)
    applied = false
    for v in vr.hard_violations, f in fixers
      occursin(f.pattern, v.constraint_name) || continue
      desc = f.transform(desc, v)
      applied = true
      break
    end
    applied || return (desc, vr)
  end
  (desc, vr)
end
