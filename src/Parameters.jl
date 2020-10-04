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
end
OptionParameter{T}() where T = OptionParameter{T}(missing)

(p::OptionParameter{T})() where T =
  ismissing(p.value) ?
    error("Parameter was not initialized") :
    p.value
(p::OptionParameter{T})(newvalue::T) where T = p.value = newvalue

mutable struct LazyParameter{T}
  initializer::Union{DataType,Function} #This should be a more specific type: None->T
  value::Union{T,Nothing}
end

LazyParameter(T::DataType, initializer::Union{DataType,Function}) =
  LazyParameter{T}(initializer, nothing)

(p::LazyParameter{T})() where T = isnothing(p.value) ? (p.value = p.initializer()) : p.value
(p::LazyParameter{T})(newvalue::T) where T = p.value = newvalue

Base.reset(p::LazyParameter{T}) where T = p.value = nothing
