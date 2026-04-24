#=
Backend API import machinery.

KhepriBase exposes two tiers of symbols:

- `export`-ed names are the portable user API (e.g., `sphere`, `xyz`, `material`).
  Anyone doing `using KhepriBackend` sees these via Interface.jl's `@reexport`.

- `public` names are the developer API used by backend implementors (e.g., `b_sphere`,
  `b_trig`, `realize`, `NativeRef`, internal macros). These are NOT re-exported to
  end users, but backend packages need them in scope to define methods on them.

`@import_backend_api` bridges the gap: when invoked from a backend's Interface.jl,
it splices in a single `import KhepriBase: n1, n2, ...` pulling every public-but-not-
exported name into the backend module. `import` (not `using`) is required because
backends extend these functions with new methods (e.g. `b_sphere(b::MyBackend, ...)
 = ...`); Julia 1.11+ disallows extending names brought in by `using M: x`.
Developers get the full low-level surface without having to maintain an import list.

The list is computed at KhepriBase load time (this file is included last) via
introspection of `names(KhepriBase; all=true)` filtered by `Base.ispublic` and
`!Base.isexported`.
=#

const _BACKEND_API_NAMES = let m = @__MODULE__
  filter(names(m; all=true)) do s
    str = string(s)
    !startswith(str, "#") &&
      s !== :_BACKEND_API_NAMES &&
      isdefined(m, s) &&
      Base.ispublic(m, s) &&
      !Base.isexported(m, s)
  end
end

macro import_backend_api()
  isempty(_BACKEND_API_NAMES) && return :()
  Expr(:import,
    Expr(:(:),
      Expr(:(.), :KhepriBase),
      (Expr(:(.), s) for s in _BACKEND_API_NAMES)...))
end

public _BACKEND_API_NAMES, @import_backend_api
