export random, random_range, random_seed, set_random_seed

const random_seed = Parameter(12345)

set_random_seed(v::Int) = random_seed(v)

next_random(previous_random::Int) =
  let test = 16807*rem(previous_random,127773) - 2836*div(previous_random,127773)
    if test > 0
      if test > 2147483647
        test - 2147483647
      else
        test
      end
    else
      test + 2147483647
    end
  end

next_random!() = random_seed(next_random(random_seed()))

random(x::Int) = rem(next_random!(), x)

random(x::Real) = x*next_random!()/2147483647.0

random_range(x0, x1) =
  if x0 == x1
    x0
  else
    x0 + random(x1 - x0)
  end

#########################################

export RGB, rgb, rgba, rgb_radiance
const rgb = RGB
const rgba = RGBA

rgb_radiance(c::RGB) = 0.265*red(c)+0.67*green(c)+0.065*blue(c)

#########################################

required() = error("Required parameter")

#########################################
export division, map_division

division(t0, t1, n::Int, include_last::Bool=true) =
  let n = convert(Int, n), iter = range(t0, stop=t1, length=n + 1)
    collect(include_last ? iter : take(iter, n))
  end

# Generic objects are processed using map_division
division(obj::Any, n::Int) = map_division(identity, obj, n)


map_division(f, t0, t1, n::Int, include_last::Bool=true) =
  let n = convert(Int, n), iter = range(t0, stop=t1, length=n + 1)
    map(f, include_last ? iter : take(iter, n))
  end

map_division(f, u0, u1, nu::Int, include_last_u::Bool, v0, v1, nv::Int) =
  map_division(u -> map_division(v -> f(u, v), v0, v1, nv),
               u0, u1, nu, include_last_u)

map_division(f, u0, u1, nu::Int, v0, v1, nv::Int, include_last_v::Bool=true) =
  map_division(u -> map_division(v -> f(u, v), v0, v1, nv, include_last_v),
               u0, u1, nu)

map_division(f, u0, u1, nu::Int, include_last_u::Bool, v0, v1, nv::Int, include_last_v::Bool) =
  map_division(u -> map_division(v -> f(u, v), v0, v1, nv, include_last_v),
               u0, u1, nu, include_last_u)

########################################
# Grasshopper compatibility

export series, crossref, remap, cull

series(start::Real, step::Real, count::Int) =
  range(start, step=step, length=count)

export crossref_holistic
crossref_holistic(arr1, arr2) =
  vcat([arr1[i] for i in range(1, stop=length(arr1)) for j in arr2]...),
  vcat([arr2 for i in arr1]...)

crossref(as, bs) = [(a, b) for a in as, b in bs]

remap(in, (min_in, max_in), (min_out, max_out)) =
  min_out + (max_out-min_out)/(max_in-min_in)*(in-min_in)

cull(template, as) =
  [a for (a, t) in zip(as, cycle(template)) if t]

# To create paths from paths
export path_replace_suffix
path_replace_suffix(path::String, suffix::String) =
  let (base, _) = splitext(path)
    base * suffix
  end

#

replace_in(expr::Expr, replacements) =
    if expr.head == :.
        Expr(expr.head,
             replace_in(expr.args[1], replacements), expr.args[2])
    elseif expr.head == :quote
        expr
    else
        Expr(expr.head,
             map(arg -> replace_in(arg, replacements), expr.args) ...)
    end
replace_in(expr::Symbol, replacements) =
    get(replacements, expr, esc(expr))
replace_in(expr::Any, replacements) =
    expr

function process_named_params(def)
  call, body = def.args[1], def.args[2]
  name, params = call.args[1], call.args[2:end]
  idx = findfirst(p->p isa Expr && p.head==:kw, params)
  mand, opts = isnothing(idx) ? (params, []) : (params[1:idx-1], params[idx:end])
  opt_names = map(opt -> opt.args[1] isa Expr && opt.args[1].head==:(::) ? opt.args[1].args[1] : opt.args[1], opts)
  opt_types = map(opt -> opt.args[1] isa Expr && opt.args[1].head==:(::) ? opt.args[1].args[2] : :Any, opts)
  opt_inits = map(opt -> opt.args[2], opts)
  mk_param(name,typ,init) = Expr(:kw, Expr(:(::), name, esc(typ)), init)
  opt_params = map(mk_param, opt_names, opt_types, opt_inits)
  key_params = map(mk_param, opt_names, opt_types, opt_names)
  # protect stuff
  name = esc(name)
  mand = map(esc, mand)
  body = esc(body)
  :($(name)($(mand...), $(opt_params...); $(key_params...)) = $(body))
end

macro named_params(def)
  process_named_params(def)
end

#=
@macroexpand @named_params bar() = 1
@macroexpand @named_params bar(a) = a
@macroexpand @named_params bar(a::Real) = a
@macroexpand @named_params bar(a=1) = a
@macroexpand @named_params bar(a::Real=1) = a
@macroexpand @named_params bar(a::Int, b::Real) = a + b
@macroexpand @named_params bar(a::Int, b::Real=3) = a + b
@macroexpand @named_params bar(a::Int, b::Real=x+3) = a + b
@macroexpand @named_params bar(a::Int=2, b::Real=3) = a + b
=#

export reverse_dict
reverse_dict(dict) =
  let rev_dict = Dict()
    for k in keys(dict)
      if dict[k] in keys(rev_dict)
         push!(rev_dict[dict[k]], k)
      else
         rev_dict[dict[k]] = [k]
      end
    end
  rev_dict
end

# To compute the sun's altitude and azimuth
export sun_pos
# Based on "How to compute planetary positions" by Paul Schlyter, Stockholm, Sweden
function sun_pos(year, month, day, hour, minute, Lstm, latitude, longitude)
  if abs(longitude-Lstm)>30
     @info("Longitude $(longitude) differs by more than 30 degrees from timezone meridian $(Lstm).")
  end
  # Calculate universal time (utime)
  utime = hour+(minute/60)-Lstm/15;
  if utime < 0
     day = day-1;
     utime = utime+24;
  end
  if utime > 24
     day = day+1;
     utime = utime-24;
  end
  int(x) = floor(Int, x)
  degrees(x) = 180*x/pi
  # Amount of days to, or from, the year 2000
  d = 367*year-int((7*int((year+int((month+9))/12)))/4)+int((275*month)/9)+day-730530+utime/24;
  # Longitude of perihelion (w), eccentricity (e)
  w = 282.9404+4.70935E-5*d;
  e = 0.016709-1.151E-9*d;
  mean_anomaly = 356.0470+0.9856002585*d;
  sun_longitude = w + mean_anomaly;
  # Obliquity of the ecliptic, eccentric anomaly (E)
  oblecl = 23.4393-3.563E-7*d;
  E = mean_anomaly+(180/pi)*e*sind(mean_anomaly)*(1+e*cosd(mean_anomaly));
  # Sun's rectangular coordinates in the plane of ecliptic (A,B)
  A = cosd(E)-e;
  B = sind(E)*sqrt(1-e*e);
  # Distance (r), longitude of the sun (lon)
  r = sqrt(A*A+B*B);
  true_anomaly = degrees(atan(B,A));
  lon = true_anomaly + w;
  # Calculate declination and right ascension
  decl = asin(sind(oblecl)*sind(lon));
  RA = degrees(atan(sind(lon)*cosd(oblecl),cosd(lon)))/15;
  # Greenwich meridian siderial time at 00:00 (GMST0),siderial time (SIDTIME), hour angle (HA)
  GMST0 = sun_longitude/15+12;
  SIDTIME = GMST0+utime+longitude/15;
  HA = (SIDTIME-RA)*15;
  # This is what we're looking for: Altitude & Azimuth
  Al = degrees(asin(sind(latitude)*sin(decl)+cosd(latitude)*cos(decl)*cosd(HA)));
  Az = degrees(atan(sind(HA),cosd(HA)*sind(latitude)-tan(decl)*cosd(latitude)))+180;
  Al, Az
end

sun_pos(date, timezone, latitude, longitude) =
  sun_pos(year(date), month(date), day(date), hour(date), minute(date), timezone, latitude, longitude)

# Lists

export List, Nil, Cons, list, cons, nil, head, tail
abstract type List{T} end
struct Nil{T} <: List{T} end
struct Cons{T} <: List{T}
  head::T
  tail::List{T}
end
Base.Pair(h::T, t::List{L}) where {T, L} = Cons{typejoin(T, L)}(h, t)
nil = Nil{Union{}}()
cons(h::T, t::List{L}) where {T, L} = Cons{typejoin(T, L)}(h, t)
(::Colon)(h::T, t::List{L}) where {T, L} = Cons{typejoin(T, L)}(h, t)
#Base.convert(::Type{List{T}}, l::Nil{Union{}}) where {T} = Nil{T}()
Base.convert(::Type{List{T1}}, l::Nil{T2}) where {T1, T2 <: T1} = Nil{T1}()

list() = nil
Base.isempty(lst::Nil) = true
Base.isempty(lst::Cons) = false
Base.firstindex(lst::List{T}) where T = 1
Base.getindex(lst::Nil, i) = throw(BoundsError(lst, i))
Base.getindex(lst::Cons, i) = i == 1 ? lst.head : getindex(lst.tail, i-1)
Base.eltype(lst::List{T}) where T = T

list(elts...) =
  let l = nil
    for i = length(elts):-1:1
      l = cons(elts[i], l)
    end
    l
  end

list(gen::Base.Generator) =
  let iter(next) =
        isnothing(next) ?
          nil :
          let (e, state) = next
            cons(e, iter(iterate(gen, state)))
          end
    iter(iterate(gen))
  end


head(x::Cons{T}) where {T} = x.head::T
tail(x::Cons{T}) where {T} = x.tail
Base.first(x::Cons) = x.head
Base.Iterators.drop(x::Cons, n::Integer) = n == 0 ? x : drop(x.tail, n-1)
Base.iterate(l::List, ::Nil) = nothing
Base.iterate(l::List, state::Cons = l) = state.head, state.tail

import Base.==
==(x::Nil, y::Nil) = true
==(x::Cons, y::Cons) = (x.head == y.head) && (x.tail == y.tail)

import Base.show
Base.show(io::IO, lst::Nil) = print(io, "List()")
Base.show(io::IO, lst::Cons{T}) where T =
  begin
    print(io, "List{$T}(")
    show(io, head(lst))
    for e in tail(lst)
      print(io, ", ")
      show(io, e)
    end
    print(io, ")")
  end

Base.length(l::Nil) = 0
Base.length(l::Cons) =
  let n = 0
    for i in l
      n += 1
    end
    n
  end

Base.map(f::Base.Callable, lst::List) = list(f(e) for e in lst)
Base.filter(f::Function, lst::List) = list(e for e in lst if f(e))

# This amounts to type piracy, so let's avoid it.
#Base.cat() = list()
Base.cat(lst::List, lsts::List...) =
  let T = typeof(lst).parameters[1]
    n = length(lst)
    for l in lsts
      T2 = typeof(l).parameters[1]
      T = typejoin(T, T2)
      n += length(l)
    end
    elems = Vector{T}(undef, n)
    i = 1
    for e in lst
      elems[i] = e
      i += 1
    end
    for lst in lsts
      for e in lst
      elems[i] = e
      i += 1
      end
    end
    let l = nil(T)
      for i = i-1:-1:1
        l = cons(elems[i], l)
      end
      l
    end
  end

# Lists can be converted to Arrays

Base.convert(::Type{Array{S,1}}, l::List{T}) where {S, T <: S} = collect(T, l)

# We need a functional way of creating shallow copies of structs where some fields are changed.
copy_with(t::T; kwargs... ) where T =
  let fieldnames = fieldnames(T), 
      nt = NamedTuple{fieldnames, Tuple{fieldtypes(T)...}}([getfield(t, name) for name in fieldnames])
    T(; merge(nt, kwargs.data)...)
end


# Different outputs can be generated from the backends
# but we can integrate them with the Julia display system.
export PNGFile, PDFFile

struct PNGFile path end
struct PDFFile path end

Base.show(io::IO, ::MIME"image/png", f::PNGFile) =
  write(io, read(f.path))

# Copyright TikzPictures
#------------------------
const tikz_id = Parameter{Int}(round(UInt64, time() * 1e6))

fixing_svg(io::IO, svgpath) =
  let s = read(svgpath, String),
      _tikzid = tikz_id()
    s = replace(s, "glyph" => "glyph-$(_tikzid)-")
    s = replace(s, "\"clip" => "\"clip-$(_tikzid)-")
    s = replace(s, "#clip" => "#clip-$(_tikzid)-")
    s = replace(s, "\"image" => "\"image-$(_tikzid)-")
    s = replace(s, "#image" => "#image-$(_tikzid)-")
    s = replace(s, "linearGradient id=\"linear" => "linearGradient id=\"linear-$(_tikzid)-")
    s = replace(s, "#linear" => "#linear-$(_tikzid)-")
    s = replace(s, "image id=\"" => "image style=\"image-rendering: pixelated;\" id=\"")
    tikz_id(_tikzid + 1)
    println(io, s)
  end

Base.show(io::IO, ::MIME"image/svg+xml", f::PDFFile) =
  let path = f.path
    ! isfile(path) ?
      error("Inexisting file $path") :
      let svgpath = path_replace_suffix(path, ".svg"),
          needs_update = ! isfile(svgpath) || mtime(path) > mtime(svgpath)
        if needs_update
          let pdftocairo = Sys.which("pdftocairo")
            if isnothing(pdftocairo)
              error("Could not find pdftocairo. Do you have MikTeX installed?")
            else
              try
                run(pipeline(`$(pdftocairo) -svg -l 1 $(path) $(svgpath)`, stdout=devnull, stderr=devnull), wait=true)
              catch e
                error("Could not process $path to generate $svgpath.")
              end
            end
          end
        end
        fixing_svg(io, svgpath)
      end
  end

  