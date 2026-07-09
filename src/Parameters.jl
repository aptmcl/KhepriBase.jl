export with, Parameter, OptionParameter, LazyParameter, ThreadLocalParameter, GlobalParameter, spawn_inheriting

#=
Parameter uses task-local storage so that each @spawn-ed task (e.g., each
WebSocket client handler) gets its own value, preventing cross-client
contamination.  The `value` field serves as the global default, returned
when the current task has not set a task-local override.
=#
mutable struct Parameter{T}
  value::T
end

# NOTE: task-local storage is NOT inherited by Threads.@spawn/@async children — a spawned task starts
# with empty TLS and sees only `value` (the global default). To build geometry in a child task under an
# enclosing `with(current_cs, cs) do … end`, use `spawn_inheriting` (below), never a bare @spawn.
(p::Parameter{T})() where T = get(task_local_storage(), p, p.value)::T
(p::Parameter{T})(newvalue::T) where T = task_local_storage(p, newvalue)::T

function with(f, p, newvalue)
  oldvalue = p()
  p(newvalue)
  try
    f()
  finally
    p(oldvalue)
  end
end

with(f, p, newvalue, others...) =
  with(p, newvalue) do
    with(f, others...)
  end

"""
    spawn_inheriting(f)

Run `f` in a freshly `Threads.@spawn`-ed task that INHERITS the caller's `Parameter` bindings.
Plain `@spawn`/`@async` children start with empty task-local storage, so `Parameter`s
(`current_cs`, `current_backends`, `default_wall_family`, …) would silently revert to their
global defaults — geometry built there would land in `world_cs`. Use this whenever a spawned task
must build geometry under an enclosing `with(current_cs, cs) do … end`.
"""
spawn_inheriting(f) =
  let parent_tls = copy(task_local_storage())
    Threads.@spawn begin
      merge!(task_local_storage(), parent_tls)
      f()
    end
  end

#=
A GlobalParameter has the old Parameter semantics: a single mutable value
shared across all tasks.  Use this for infrastructure state that is set once
(e.g., server host/port) or that must be visible across tasks (e.g.,
main_callback set in the REPL and read by server tasks).
=#
mutable struct GlobalParameter{T}
  value::T
end

(p::GlobalParameter{T})() where T = p.value::T
(p::GlobalParameter{T})(newvalue::T) where T = p.value = newvalue::T

#=
An OptionParameter is one which can be missing. Retrieving the value of a missing OptionParameter triggers an error.
Uses task-local storage like Parameter.
=#
mutable struct OptionParameter{T}
  value::Union{Missing,T}
  OptionParameter{T}(v::T1) where {T,T1<:T} = new{T}(v)
  OptionParameter{T}() where T = new{T}(missing)
  OptionParameter(v::T) where T = new{T}(v)
end

(p::OptionParameter{T})() where T =
  let v = get(task_local_storage(), p, p.value)
    ismissing(v) ?
      error("Parameter was not initialized with value of type $T") :
      v::T
  end
(p::OptionParameter{T})(newvalue::T) where T = task_local_storage(p, newvalue)::T

#=
A LazyParameter is one which is initialized only when first requested.
LazyParameter remains global (not task-local) — it is used for singletons
like the WebSocket server instance.
=#

mutable struct LazyParameter{T,F<:Function}
  initializer::F
  value::Union{T,Nothing}
  lock::ReentrantLock
  LazyParameter(initializer::F) where {F<:Function} =
    new{Base.return_types(initializer, ())[1],F}(initializer, nothing, ReentrantLock())
end

# Double-checked locking: concurrent tasks initialize the singleton exactly once, while the common
# already-initialized path stays lock-free.
(p::LazyParameter{T,F})() where {T,F} =
  isnothing(p.value) ?
    lock(p.lock) do
      isnothing(p.value) ? (p.value = p.initializer()::T) : p.value
    end :
    p.value
(p::LazyParameter{T,F})(newvalue::T) where {T,F} = lock(p.lock) do; p.value = newvalue end

Base.reset(p::LazyParameter{T,F}) where {T,F} = lock(p.lock) do; p.value = nothing end

#=
ThreadLocalParameter is now equivalent to Parameter (which uses task-local
storage).  Kept as an alias for backwards compatibility.
=#
const ThreadLocalParameter = Parameter
