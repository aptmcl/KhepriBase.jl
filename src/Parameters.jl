export with, Parameter, OptionParameter, LazyParameter

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

mutable struct LazyParameter{T,F<:Function}
  initializer::F
  value::Union{T,Nothing}
  LazyParameter(initializer::F) where {F<:Function} =
    new{Base.return_types(initializer, ())[1],F}(initializer, nothing)
end

(p::LazyParameter{T,F})() where {T,F} = isnothing(p.value) ? (p.value = p.initializer()::T) : p.value
(p::LazyParameter{T,F})(newvalue::T) where {T,F} = p.value = newvalue

Base.reset(p::LazyParameter{T,F}) where {T,F} = p.value = nothing
