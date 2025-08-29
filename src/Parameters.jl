export with, Parameter, OptionParameter, LazyParameter, ThreadLocalParameter

mutable struct Parameter{T}
  value::T
end

(p::Parameter{T})() where T = p.value::T
(p::Parameter{T})(newvalue::T) where T = p.value = newvalue::T

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
An OptionParameter is one which can be missing. Retrieving the value of a missing OptionParameter triggers an error.
=#
mutable struct OptionParameter{T}
  value::Union{Missing,T}
  OptionParameter{T}(v::T1) where {T,T1<:T} = new{T}(v)
  OptionParameter{T}() where T = new{T}(missing)
  OptionParameter(v::T) where T = new{T}(v)
end

(p::OptionParameter{T})() where T =
  ismissing(p.value) ?
    error("Parameter was not initialized with value of type $T") :
    p.value::T
(p::OptionParameter{T})(newvalue::T) where T = p.value = newvalue

#=
A LazyParameter is one which is initialized only when first requested.
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
Khepri is becoming multithreaded. This means that Parameters are no longer a good idea,
as different threads might interfere, where one changes, e.g., the current_cs and another
becomes affected. Moreover, if both try to change it, read-modify-write problems might occur.

To solve this problem, a ThreadLocalParameter is a variant of a Parameter that operates
as a thread-aware dynamic variable. There is a global value, which can be consulted and/or
updated on each thread, without interfeering with other threads.
=#

struct ThreadLocalParameter{T}
  value::T
end

(p::ThreadLocalParameter{T})() where T = get(task_local_storage(), p, p.value)::T
(p::ThreadLocalParameter{T})(newvalue::T) where T = task_local_storage(p, newvalue)::T
