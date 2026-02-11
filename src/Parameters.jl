export with, Parameter, OptionParameter, LazyParameter, ThreadLocalParameter, GlobalParameter

#=
Parameter uses task-local storage so that each @spawn-ed task (e.g., each
WebSocket client handler) gets its own value, preventing cross-client
contamination.  The `value` field serves as the global default, returned
when the current task has not set a task-local override.
=#
mutable struct Parameter{T}
  value::T
end

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
  LazyParameter(initializer::F) where {F<:Function} =
    new{Base.return_types(initializer, ())[1],F}(initializer, nothing)
end

(p::LazyParameter{T,F})() where {T,F} = isnothing(p.value) ? (p.value = p.initializer()::T) : p.value
(p::LazyParameter{T,F})(newvalue::T) where {T,F} = p.value = newvalue

Base.reset(p::LazyParameter{T,F}) where {T,F} = p.value = nothing

#=
ThreadLocalParameter is now equivalent to Parameter (which uses task-local
storage).  Kept as an alias for backwards compatibility.
=#
const ThreadLocalParameter = Parameter
