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

division(t0, t1, n::Real, include_last::Bool=true) =
  let n = convert(Int, n), iter = range(t0, stop=t1, length=n + 1)
    collect(include_last ? iter : take(iter, n))
  end

# Generic objects are processed using map_division
division(obj::Any, n::Real) = map_division(identity, obj, n)


map_division(f, t0, t1, n::Real, include_last::Bool=true) =
  let n = convert(Int, n), iter = range(t0, stop=t1, length=n + 1)
    map(f, include_last ? iter : take(iter, n))
  end

map_division(f, u0, u1, nu::Real, include_last_u::Bool, v0, v1, nv::Real) =
  map_division(u -> map_division(v -> f(u, v), v0, v1, nv),
               u0, u1, nu, include_last_u)

map_division(f, u0, u1, nu::Real, v0, v1, nv::Real, include_last_v::Bool=true) =
  map_division(u -> map_division(v -> f(u, v), v0, v1, nv, include_last_v),
               u0, u1, nu)

map_division(f, u0, u1, nu::Real, include_last_u::Bool, v0, v1, nv::Real, include_last_v::Bool) =
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
  let (base, old_suffix) = splitext(path)
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
  idx = findfirst(p->p.head==:(::), params)
  mand, opts = isnothing(idx) ? ([], params) : (esc.(params[1:idx]), params[idx+1:end])
  opt_names = map(opt -> opt.args[1].args[1], opts)
  opt_types = map(opt -> esc(opt.args[1].args[2]), opts)
  opt_inits = map(opt -> opt.args[2], opts)
  opt_renames = map(Symbol ∘ string, opt_names)
  opt_replacements = Dict(zip(opt_names, opt_renames))
  mk_param(name,typ,init) = Expr(:kw, Expr(:(::), name, typ), init)
  opt_params = map(mk_param, opt_renames, opt_types, map(init -> replace_in(init, opt_replacements), opt_inits))
  key_params = map(mk_param, opt_names, opt_types, opt_renames)
  func_name = esc(name)
  :($(func_name)($(mand...), $(opt_params...); $(key_params...)) = $(esc(body)))
end

macro named_params(def)
  process_named_params(def)
end

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
