# RPCConformanceTests.jl - Static RPC-conformance test for Khepri socket backends
#
#= Why this test exists
   --------------------
   A socket/websocket backend adapter invokes a remote plugin method through the
   `@remote(b, Name(args...))` macro (KhepriBase/src/Backends.jl), which expands to

       call_remote(getfield(getfield(b, :remote), :Name), connection(b), args...)

   `getfield(b, :remote)` is a NamedTuple keyed by RPC name, built from the
   `xxx_api = @remote_api :LANG """ ...signatures... """` block. If `Name` is NOT
   a key of that NamedTuple, the call throws

       type NamedTuple has no field Name

   on its FIRST invocation — a runtime crash that only surfaces when that code path
   is exercised against a live CAD app. Past instances of this bug class include
   Revit Box/Pyramid, Rhino SpotLight/IESLight, and Unity StartSelectingPosition.

   This test catches the whole class STATICALLY, with no CAD connection: it reads the
   declared RPC names from the loaded backend instance's `:remote` field, AST-walks
   every source file for `@remote`/`@get_remote` call heads, and asserts the called
   set is a subset of the declared set. It is intended to run unconditionally in CI,
   alongside (not gated like) the connection-dependent visual/stress tests.

   See also: BackendConformanceTests.jl (the b_* contract suite, which DOES need a
   live backend), VisualTests.jl. =#
#
# Usage (from a backend's runtests.jl, unconditionally):
#   include(joinpath(dirname(pathof(KhepriBase)), "..", "test", "RPCConformanceTests.jl"))
#   using .RPCConformanceTests
#   run_rpc_conformance_tests(rhino, joinpath(dirname(pathof(KhepriRhino))))

module RPCConformanceTests

using Test
using KhepriBase
KhepriBase.@import_backend_api

export run_rpc_conformance_tests

#= Which macros name an RPC.
   `@remote(b, Name(args...))` -> RPC name is the head symbol `Name` of the call.
   `@get_remote(b, Name)`      -> RPC name is the bare symbol `Name`.
   We key off the MACRO HEAD, never the backend-argument name: some call sites use
   the literal instance (e.g. `@remote(rhino, RunGrasshopperSolver())`) instead of
   the conventional `b`, so matching on the first argument would miss them. =#
const REMOTE_MACROS = Set([Symbol("@remote"), Symbol("@get_remote")])

#= Resolve the head symbol of a macro's RPC payload.
   The payload is `Name(args...)` (an `:call` Expr, head symbol = RPC name) for
   `@remote`, or a bare `Name` (a Symbol) for `@get_remote`. Anything else
   (dynamically-built names, non-call expressions) yields `nothing` and is skipped —
   this static check cannot reason about non-literal call heads. =#
rpc_name(payload::Symbol) = payload
rpc_name(payload::Expr) =
  (payload.head === :call && !isempty(payload.args) && payload.args[1] isa Symbol) ?
    payload.args[1] : nothing
rpc_name(::Any) = nothing

#= Record of one call site: the RPC name plus where it appeared.
   On a conformance failure we surface name + file + line so the offending adapter
   call can be located immediately. =#
struct CallSite
  name::Symbol
  file::String
  line::Int
end

#= AST walk collecting every @remote / @get_remote call site under an expression.
   Why an AST walk rather than a regex:
     - It is comment-safe by construction: `Meta.parseall` drops `#` line comments
       and `#= =#` block comments, so commented-out `@remote(...)` calls and
       large disabled `#= ... =#` blocks (e.g. KhepriUnity's b_polygon stub block)
       are never counted.
     - It handles multiline calls and literal-instance call sites that a
       `@remote(\s*b\s*,` regex would miss.
   `cur` tracks the most recent LineNumberNode so each CallSite gets an accurate line. =#
function walk!(ex, file::String, cur::Ref{Int}, found::Vector{CallSite})
  ex isa Expr || return found
  if ex.head === :macrocall && !isempty(ex.args) && ex.args[1] isa Symbol &&
     ex.args[1] in REMOTE_MACROS
    # The macrocall's own LineNumberNode (if present) is the most precise location.
    line = cur[]
    for a in ex.args
      a isa LineNumberNode && (line = a.line)
    end
    rest = filter(a -> !(a isa LineNumberNode), ex.args[2:end])  # [backend, payload]
    if length(rest) >= 2
      nm = rpc_name(rest[2])
      nm !== nothing && push!(found, CallSite(nm, file, line))
    end
  end
  for a in ex.args
    a isa LineNumberNode && (cur[] = a.line)
    walk!(a, file, cur, found)
  end
  found
end

#= Collect call sites from every *.jl file under `src_dir` (recursively).
   Reads source text and parses it with `Meta.parseall` — no module evaluation, so
   this works even for backends whose instance fails to load for unrelated reasons
   (the declared set still comes from a loaded instance, but extraction is textual). =#
function collect_call_sites(src_dir::AbstractString)
  found = CallSite[]
  for (root, _dirs, files) in walkdir(src_dir)
    for f in files
      endswith(f, ".jl") || continue
      path = joinpath(root, f)
      ast = Meta.parseall(read(path, String); filename = path)
      walk!(ast, path, Ref(0), found)
    end
  end
  found
end

#= The declared RPC set lives in the instance's `:remote` NamedTuple field.
   We read it via `getfield` (NOT `connection(b)`), so no live CAD connection is
   needed — the field is fully populated by `using KhepriXXX`. Works identically for
   SocketBackend and WebSocketBackend (both carry a `:remote` NamedTuple). =#
declared_rpc_names(b) = Set(keys(getfield(b, :remote)))

"""
    run_rpc_conformance_tests(b::Backend, src_dir; skip=Symbol[])

Statically verify that every RPC the backend adapter calls via `@remote` /
`@get_remote` is declared in the backend's `@remote_api` block.

`b` is a loaded backend instance (e.g. `rhino`); `src_dir` is the package source
directory, typically `joinpath(dirname(pathof(KhepriXXX)))`. Names in `skip` are
exempted from the subset assertion (escape hatch for intentionally-dynamic or
otherwise known-acceptable calls).

Requires no live connection: only `getfield(b, :remote)` (populated by `using`) and
source-file reads are performed.
"""
function run_rpc_conformance_tests(b::Backend, src_dir::AbstractString;
                                   skip::Vector{Symbol} = Symbol[])
  isdir(src_dir) ||
    error("run_rpc_conformance_tests: src_dir does not exist: $(src_dir)")
  declared = declared_rpc_names(b)
  skipset = Set(skip)
  sites = collect_call_sites(src_dir)

  # One representative call site per undeclared name keeps failure output readable.
  offenders = CallSite[]
  seen = Set{Symbol}()
  for s in sites
    (s.name in declared || s.name in skipset || s.name in seen) && continue
    push!(seen, s.name)
    push!(offenders, s)
  end
  sort!(offenders, by = s -> string(s.name))

  @testset "RPC Conformance: $(backend_name(b))" begin
    if !isempty(offenders)
      @info "Undeclared RPC calls in $(backend_name(b)) " *
            "(called via @remote/@get_remote but absent from the @remote_api NamedTuple):"
      for s in offenders
        @info "  $(s.name)  —  $(s.file):$(s.line)"
      end
    end
    # The whole class reduces to: every called name must be a declared key.
    # On failure, @test shows the offending-name vector so the report is actionable
    # even without scrolling back to the @info lines above.
    offending_names = [s.name for s in offenders]
    @test offending_names == Symbol[]
  end
end

end # module RPCConformanceTests
