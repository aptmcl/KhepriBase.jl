export with, Parameter, LazyParameter

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

mutable struct LazyParameter{T}
  initializer::Union{DataType,Function} #This should be a more specific type: None->T
  value::Union{T,Nothing}
end

LazyParameter(T::DataType, initializer::Union{DataType,Function}) =
  LazyParameter{T}(initializer, nothing)

(p::LazyParameter{T})() where T = isnothing(p.value) ? (p.value = p.initializer()) : p.value
(p::LazyParameter{T})(newvalue::T) where T = p.value = newvalue

Base.reset(p::LazyParameter{T}) where T = p.value = nothing
