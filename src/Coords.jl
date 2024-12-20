export X, XY, XYZ, Pol, Pold, Cyl, Sph,
       VX, VXY, VXYZ, VPol, VPold, VCyl, VSph,
       xyz, cyl, sph,
       vxyz, vcyl, vsph,
       world_cs,
       current_cs,
       distance,
       u0,ux,uy,uz,uxy,uyz,uxz,uxyz,
       x,y,z,
       xy,xz,yz,pol,cyl,sph,
       cx,cy,cz,
       pol_rho, pol_phi,
       cyl_rho,cyl_phi,cyl_z,
       sph_rho,sph_phi,sph_psi,
       uvx,uvy,uvz,uvxy,uvyz,uvxz,uvxyz,
       vx,vy,vz,
       vxy,vxz,vyz,vpol,vcyl,vsph,
       add_x,add_y,add_z,add_xy,add_xz,add_yz,add_xyz,
       add_pol,add_cyl,add_sph,
       unitized, dot, cross,
       cs_from_o_vx_vy_vz,
       cs_from_o_vx_vy,
       cs_from_o_vz,
       cs_from_o_phi,
       cs_from_o_rot_x,
       cs_from_o_rot_y,
       cs_from_o_rot_z,
       cs_from_o_rot_zyx,
       loc_from_o_vx,
       loc_from_o_vx_vy,
       loc_from_o_vz,
       loc_from_o_phi,
       loc_from_o_rot_x,
       loc_from_o_rot_y,
       loc_from_o_rot_z,
       loc_from_o_rot_zyx,
       min_loc, max_loc,
       is_world_cs,
       in_cs, in_world,
       on_cs,
       intermediate_loc,
       meta_program,
       translated_cs,
       scaled_cs,
       center_scaled_cs,
       translating_current_cs,
       regular_polygon_vertices,
       norm,
       angle_between,
       rotate_vector,
       raw_point,
       raw_plane

#=
Some useful terminology:

Coordinate Space (also known as Reference Frame): an origin (position) and
three axis (directions) that allow the precise identification of locations and
translations (which includes directions and magnitude). There can be many reference
frames. One is considered the world reference frame. It is possible to interpret
the same location or the same translation regarding different reference frames.

Coordinate System: a way to assign meaning to numbers that represent locations
or translations relative to a reference frame. Different coordinate systems (e.g.,
rectangular, polar, cylindrical, and spherical) assign different meanings to the
three numbers. It is possible to convert between different coordinate systems.

Location: represented by a triple of numbers using a Coordinate System and a
Reference Frame.

Translation: represented by a triple of numbers using a Coordinate System and a
Reference Frame.

We need to frequently add translations to locations but it might happen that these
translations and locations have different reference frames, which cause surprising
results. To avoid this problem, we introduce the concept of void reference frame,
a reference frame which does not

=#

const Vec4f = SVector{4,Float64}
const Mat4f = SMatrix{4,4,Float64}

struct CS
  transform::Mat4f
#  invtransform::Mat4f
end

show(io::IO, cs::CS) =
  print(io, cs == world_cs ? "world_cs" : "..._cs")

translated_cs(cs::CS, x::Real, y::Real, z::Real) =
  CS(cs.transform * Mat4f([
      1 0 0 x;
      0 1 0 y;
      0 0 1 z;
      0 0 0 1]))

scaled_cs(cs::CS, x::Real, y::Real, z::Real) =
  CS(cs.transform * Mat4f([
      x 0 0 0;
      0 y 0 0;
      0 0 z 0;
      0 0 0 1]))

rotated_x_cs(cs::CS, α::Real) =
  let (s, c) = sincos(α)
    CS(cs.transform * Mat4f([
        1 0  0 0;
        0 c -s 0;
        0 s  c 0;
        0 0  0 1]))
  end
rotated_y_cs(cs::CS, α::Real) =
  let (s, c) = sincos(α)
    CS(cs.transform * Mat4f([
        c 0 s 0;
        0 1 0 0;
       -s 0 c 0;
        0 0 0 1]))
  end
rotated_z_cs(cs::CS, α::Real) =
  let (s, c) = sincos(α)
    CS(cs.transform * Mat4f([
        c -s 0 0;
        s  c 0 0;
        0  0 1 0;
        0  0 0 1]))
  end
rotated_zyx_cs(cs::CS, x::Real, y::Real, z::Real) =
  rotated_x_cs(rotated_y_cs(rotated_z_cs(cs, z), y), x)

export rotated_x_cs, rotated_y_cs, rotated_z_cs, rotated_zyx_cs


center_scaled_cs(cs::CS, x::Real, y::Real, z::Real) =
    let xt = cs.transform[4,1]
        yt = cs.transform[4,2]
        zt = cs.transform[4,3]
        translated_cs(
            scaled_cs(
                translated_cs(cs, -xt, -yt, -zt),
                x, y, z),
            xt, yt, zt)
    end

export rotated_around_p_v_cs
rotated_around_p_v_cs(cs::CS, a::Real, b::Real, c::Real, u::Real, v::Real, w::Real, ϕ::Real) =
  let u2 = u*u,
      v2 = v*v,
      w2 = w*w,
      cosT = cos(ϕ),
      oneMinusCosT = 1-cosT,
      sinT = sin(ϕ),
      m11 = u2 + (v2 + w2) * cosT,
      m12 = u*v * oneMinusCosT - w*sinT,
      m13 = u*w * oneMinusCosT + v*sinT,
      m14 = (a*(v2 + w2) - u*(b*v + c*w))*oneMinusCosT + (b*w - c*v)*sinT,
      m21 = u*v * oneMinusCosT + w*sinT,
      m22 = v2 + (u2 + w2) * cosT,
      m23 = v*w * oneMinusCosT - u*sinT,
      m24 = (b*(u2 + w2) - v*(a*u + c*w))*oneMinusCosT + (c*u - a*w)*sinT,
      m31 = u*w * oneMinusCosT - v*sinT,
      m32 = v*w * oneMinusCosT + u*sinT,
      m33 = w2 + (u2 + v2) * cosT,
      m34 = (c*(u2 + v2) - w*(a*u + b*v))*oneMinusCosT + (a*v - b*u)*sinT
    CS(cs.transform * Mat4f([
          m11 m12 m13 m14;
          m21 m22 m23 m24;
          m31 m32 m33 m32;
          0 0 0 1]))
  end

rotate_vector(vector, axis, angle) =
  let s = sin(angle),
      c = cos(angle),
      x = (c + axis.x^2*(1 - c))*vector.x +
          (axis.x*axis.y*(1 - c) - axis.z*s)*vector.y +
          (axis.x*axis.z*(1 - c) + axis.y*s)*vector.z,
      y = (axis.y*axis.x*(1 - c) + axis.z*s)*vector.x +
          (c + axis.y^2*(1 - c))*vector.y +
          (axis.y*axis.z*(1 - c) - axis.x*s)*vector.z,
      z = (axis.z*axis.x*(1 - c) - axis.y*s)*vector.x +
          (axis.z*axis.y*(1 - c) + axis.x*s)*vector.y +
          (c + axis.z^2*(1 - c))*vector.z
    vxyz(x,y,z)
  end


const world_cs = CS(Mat4f(I))
const current_cs = Parameter(world_cs)
# Special cs for "transparent" vectors
const null_cs = CS(Mat4f(I))

is_world_cs(cs::CS) = cs ===  world_cs

translating_current_cs(f, _dx::Real=0, _dy::Real=0, _dz::Real=0; dx::Real=_dx, dy::Real=_dy, dz::Real=_dz) =
    with(current_cs, translated_cs(current_cs(), dx, dy, dz)) do
        f()
    end

macro coord_structs(T, cs...)
  let gen_struct(T, ST, cs) =
        quote
            struct $T <: $ST
                    $([:($c :: Real) for c in cs]...)
                    cs::CS
                    raw::Vec4f
            end
        end,
      gen_show(T, cs) =
        quote
          Base.show(io::IO, loc :: $T) = begin
            print(io, $(lowercase(string(T))), "(")
            $([i == 1 ? :(print(io, loc.$c)) : :(print(io, ", ", loc.$c))
               for (i, c) in enumerate(cs)]...) #$(loc.cs == world_cs ? "" : ", ..."))")
            if loc.cs == world_cs
              print(io, ")")
            else
              print(io, ", ...)")
            end
          end
        end,
      VT = Symbol("V", T),                   # VX, VXY, ...
      LocT = Symbol("Loc", length(cs), "D"), # Loc1D, Loc2D, ...
      VecT = Symbol("Vec", length(cs), "D")  # Vec1D, Vec2D, ...
    esc(
      quote
        $(gen_struct(T, LocT, cs))
        $(gen_show(T, cs))
        $(gen_struct(VT, VecT, cs))
        $(gen_show(VT, cs))
      end)
  end
end

@coord_structs(X, x)
@coord_structs(XY, x, y)
@coord_structs(Pol, ρ, ϕ)
@coord_structs(Pold, ρ, ϕ)
@coord_structs(XYZ, x, y, z)
@coord_structs(Cyl, ρ, ϕ, z)
@coord_structs(Sph, ρ, ϕ, ψ)

macro coord_constructs(sig, cons, exprs...)
  let c = sig.args[1],
      params = sig.args[2:end],
      C = cons.args[1],
      args = cons.args[2:end],
      gen_construct(C, c, i, v) =
        :($c($([Expr(:kw, :($p::Real), i) for p in params]...), cs::CS=current_cs()) =
            $C($(args...), cs, Vec4f($(exprs...), $v))),
      vc = Symbol("v", c), # vx, vy, ...
      VC = Symbol("V", C) # VX, VXY, ...
    esc(
      quote
        $(gen_construct(C, c, 0, 1.0))
        $(gen_construct(VC, vc, 1, 0.0))
      end)
  end
end

@coord_constructs(x(x), X(x), x, 0.0, 0.0)
@coord_constructs(xy(x, y), XY(x, y), x, y, 0.0)
@coord_constructs(y(y), XY(0, y), 0.0, y, 0.0)
@coord_constructs(pol(ρ, ϕ), Pol(ρ, ϕ), ρ*cos(ϕ), ρ*sin(ϕ), 0.0)
@coord_constructs(pold(ρ, ϕ), Pold(ρ, ϕ), ρ*cosd(ϕ), ρ*sind(ϕ), 0.0)
@coord_constructs(xyz(x, y, z), XYZ(x, y, z), x, y, z)
@coord_constructs(xz(x, z), XYZ(x, 0, z), x, 0.0, z)
@coord_constructs(yz(y, z), XYZ(0, y, z), 0.0, y, z)
@coord_constructs(z(z), XYZ(0, 0, z), 0.0, 0.0, z)
@coord_constructs(cyl(ρ, ϕ, z), Cyl(ρ, ϕ, z), ρ*cos(ϕ), ρ*sin(ϕ), z)
@coord_constructs(sph(ρ, ϕ, ψ), Sph(ρ, ϕ, ψ), ρ*cos(ϕ)*sin(ψ), ρ*sin(ϕ)*sin(ψ), ρ*cos(ψ))

# Selectors

cx(p::Union{Loc,Vec}) = p.x
cy(p::Union{Loc,Vec}) = p.y
cz(p::Union{Loc,Vec}) = p.z

sph_rho(p::Union{Loc,Vec}) =
  let (x, y, z) = (p.x, p.y, p.z)
    sqrt(x*x + y*y + z*z)
  end
sph_phi(p::Union{Loc,Vec}) =
  let (x, y) = (p.x, p.y)
    0 == x == y ? 0 : mod(atan(y, x), 2pi)
  end
sph_phi(p::Union{Pol,VPol,Cyl,VCyl,Sph,VSph}) =
  p.ϕ

sph_psi(p::Union{Loc,Vec}) =
  let (x, y, z) = (p.x, p.y, p.z)
    0 == x == y == z ? 0 : mod(atan(sqrt(x*x + y*y), z), 2pi)
  end

cyl_rho(p::Union{Loc,Vec}) =
  let (x, y) = (p.x, p.y)
    sqrt(x*x + y*y)
  end
cyl_phi(p::Union{Loc,Vec}) = sph_phi(p)
cyl_z(p::Union{Loc,Vec}) = p.z

pol_rho = cyl_rho
pol_phi = cyl_phi

Base.getproperty(s::Union{X,VX}, sym::Symbol) =
    sym === :y ? 0 :
    sym === :z ? 0 :
    sym === :ρ ? abs(s.x) :
    sym === :ϕ ? (s.x < 0 ? 1π : 0) :
    sym === :ψ ? 0 :
    getfield(s, sym)
Base.getproperty(s::Union{XY,VXY}, sym::Symbol) =
    sym === :z ? 0 :
    sym === :ρ ? (0 == s.x ? abs(s.y) : s.y == 0 ? abs(s.x) : sqrt(s.x^2 + s.y^2)) :
    sym === :ϕ ? (0 == s.x == s.y ? 0 : mod(atan(s.y, s.x), 2pi)) :
    sym === :ψ ? 0 :
    getfield(s, sym)
Base.getproperty(s::Union{XYZ,VXYZ}, sym::Symbol) =
    sym === :ρ ? sqrt(s.x^2 + s.y^2 + s.z^2) :
    sym === :ϕ ? (0 == s.x == s.y ? 0 : mod(atan(s.y, s.x), 2pi)) :
    sym === :ψ ? (0 == s.x == s.y == s.z ? 0 : mod(atan(sqrt(s.x^2 + s.y^2), s.z), 2pi)) :
    getfield(s, sym)
Base.getproperty(s::Union{Pol,VPol}, sym::Symbol) =
    sym === :x ? s.ρ*cos(s.ϕ) :
    sym === :y ? s.ρ*sin(s.ϕ) :
    sym === :z ? 0 :
    sym === :ψ ? 0 :
    getfield(s, sym)
Base.getproperty(s::Union{Cyl,VCyl}, sym::Symbol) =
    sym === :x ? s.ρ*cos(s.ϕ) :
    sym === :y ? s.ρ*sin(s.ϕ) :
    sym === :ψ ? 0 :
    getfield(s, sym)
Base.getproperty(s::Union{Sph,VSph}, sym::Symbol) =
    sym === :x ? s.ρ*cos(s.ϕ)*sin(s.ψ) :
    sym === :y ? s.ρ*sin(s.ϕ)*sin(s.ψ) :
    sym === :z ? s.ρ*cos(s.ψ) :
    getfield(s, sym)

#=
gen_addition(C, c, Vc, vc) =
  :((+)($c :: $C, $vc :: $Vc) =
      let $vc = in_cs($vc, $c.cs)
          $C($([:($c.$p + $vc.$p) for p in params]...), $c.cs)
      end),
=#
(+)(p::X, v::VX) =
    p.cs === v.cs ? x(p.x + v.x, p.cs) : p + in_cs(v, p.cs)
(+)(p::Union{X,XY}, v::Union{VX,VXY,VPol,VPold}) =
    p.cs === v.cs ? xy(p.x + v.x, p.y + v.y, p.cs) : p + in_cs(v, p.cs)
(+)(p::Union{X,XY,XYZ}, v::Union{VX,VXY,VPol,VPold,VXYZ,VCyl,VSph}) =
    p.cs === v.cs ? xyz(p.x + v.x, p.y + v.y, p.z + v.z, p.cs) : p + in_cs(v, p.cs)
(+)(p::Pol, v::Union{VX,VXY,VPol,VPold}) =
    pol(xy(p) + v)
(+)(p::Cyl, v::Union{VX,VXY,VXYZ,VPol,VPold,VCyl,VSph}) =
    cyl(xyz(p) + v)
(+)(p::Sph, v::Union{VX,VXY,VXYZ,VPol,VPold,VCyl,VSph}) =
    sph(xyz(p) + v)
(+)(v::Vec, p::Loc) = p + v

(-)(p::X, v::VX) =
    p.cs === v.cs ? x(p.x - v.x, p.cs) : p - in_cs(v, p.cs)
(-)(p::Union{X,XY}, v::Union{VX,VXY,VPol,VPold}) =
    p.cs === v.cs ? xy(p.x - v.x, p.y - v.y, p.cs) : p - in_cs(v, p.cs)
(-)(p::Union{X,XY,XYZ}, v::Union{VX,VXY,VPol,VPold,VXYZ,VCyl,VSph}) =
    p.cs === v.cs ? xyz(p.x - v.x, p.y - v.y, p.z - v.z, p.cs) : p - in_cs(v, p.cs)
(-)(p::Pol, v::Union{VX,VXY,VPol,VPold}) =
    pol(xy(p) - v)
(-)(p::Cyl, v::Union{VX,VXY,VXYZ,VPol,VPold,VSph}) =
    cyl(xyz(p) - v)
(-)(p::Sph, v::Union{VX,VXY,VXYZ,VPol,VPold,VCyl}) =
    sph(xyz(p) - v)
(-)(p::X, q::X) =
    p.cs === q.cs ? vx(p.x - q.x, p.cs) : p - in_cs(q, p.cs)
(-)(p::Union{X,XY}, q::Union{X,XY,Pol,Pold}) =
    p.cs === q.cs ? vxy(p.x - q.x, p.y - q.y, p.cs) : p - in_cs(q, p.cs)
(-)(p::Union{X,XY,XYZ}, q::Union{X,XY,XYZ,Pol,Pold,XYZ,Cyl,Sph}) =
    p.cs === q.cs ? vxyz(p.x - q.x, p.y - q.y, p.z - q.z, p.cs) : p - in_cs(q, p.cs)
(-)(p::Pol, v::Union{X,XY,Pol,Pold}) =
    vpol(xy(p) - v)
(-)(p::Cyl, v::Union{X,XY,XYZ,Pol,Pold,Cyl,Sph}) =
    vcyl(xyz(p) - v)
(-)(p::Sph, v::Union{X,XY,XYZ,Pol,Pold,Cyl,Sph}) =
    vsph(xyz(p) - v)


#
(-)(v::VX) = vx(-v.x, v.cs)
(-)(v::VXY) = vxy(-v.x, -v.y, v.cs)
(-)(v::VXYZ) = vxyz(-v.x, -v.y, -v.z, v.cs)
(-)(v::VPol) = vpol(v.ρ, v.ϕ+π)
(-)(v::VCyl) = vcyl(v.ρ, v.ϕ+π, -v.z)
(-)(v::VSph) = vsph(v.ρ, v.ϕ+π, -v.ψ)


(*)(v::VX, r::Real) =
    vx(v.x*r, v.cs)
(*)(v::VXY, r::Real) =
    vxy(v.x*r, v.y*r, v.cs)
(*)(v::VXYZ, r::Real) =
    vxyz(v.x*r, v.y*r, v.z*r, v.cs)
(*)(v::VPol, r::Real) =
    vpol(v.ρ*r, v.ϕ, v.cs)
(*)(v::VCyl, r::Real) =
    vcyl(v.ρ*r, v.ϕ, v.z, v.cs)
(*)(v::VSph, r::Real) =
    vsph(v.ρ*r, v.ϕ, v.ψ, v.cs)
#
(*)(r::Real, v::VX) =
    vx(v.x*r, v.cs)
(*)(r::Real, v::VXY) =
    vxy(v.x*r, v.y*r, v.cs)
(*)(r::Real, v::VXYZ) =
    vxyz(v.x*r, v.y*r, v.z*r, v.cs)
(*)(r::Real, v::VPol) =
    vpol(v.ρ*r, v.ϕ, v.cs)
(*)(r::Real, v::VCyl) =
    vcyl(v.ρ*r, v.ϕ, v.z, v.cs)
(*)(r::Real, v::VSph) =
    vsph(v.ρ*r, v.ϕ, v.ψ, v.cs)

xy(v::Union{X,Pol,Pold}) = xy(v.x, v.y, v.cs)
pol(v::Union{X,XY,Pold}) = pol(v.ρ, v.ϕ, v.cs)
xyz(v::Union{X,XY,Pol,Pold,Cyl,Sph}) = xyz(v.x, v.y, v.z, v.cs)
cyl(v::Union{X,XY,XYZ,Pol,Pold,Sph}) = cyl(v.ρ, v.ϕ, v.z, v.cs)
sph(v::Union{X,XY,XYZ,Pol,Pold,Cyl}) = sph(v.ρ, v.ϕ, v.ψ, v.cs)

vxy(v::Union{VX,VPol,VPold}) = vxy(v.x, v.y, v.cs)
vpol(v::Union{VX,VXY,VPold}) = vpol(v.ρ, v.ϕ, v.cs)
vcyl(v::Union{VX,VXY,VXYZ,VPol,VPold,VSph}) = vcyl(v.ρ, v.ϕ, v.z, v.cs)
vsph(v::Union{VX,VXY,VXYZ,VPol,VPold,VCyl}) = vsph(v.ρ, v.ϕ, v.ψ, v.cs)
#
(+)(p::VX, v::VX) =
    p.cs === v.cs ? vx(p.x + v.x, p.cs) : p + in_cs(v, p.cs)
(+)(p::Union{VX,VXY}, v::Union{VX,VXY,VPol,VPold}) =
    p.cs === v.cs ? vxy(p.x + v.x, p.y + v.y, p.cs) : p + in_cs(v, p.cs)
(+)(p::Union{VX,VXY,VXYZ}, v::Union{VX,VXY,VPol,VPold,VXYZ,VCyl,VSph}) =
    p.cs === v.cs ? vxyz(p.x + v.x, p.y + v.y, p.z + v.z, p.cs) : p + in_cs(v, p.cs)
(+)(p::VPol, v::Union{VX,VXY,VPol,VPold}) =
    vpol(vxy(p) + v)
(+)(p::VCyl, v::Union{VX,VXY,VXYZ,VPol,VPold,VSph}) =
    vcyl(vxyz(p) + v)
(+)(p::VSph, v::Union{VX,VXY,VXYZ,VPol,VPold,VCyl}) =
    vsph(vxyz(p) + v)

(-)(p::VX, v::VX) =
    p.cs === v.cs ? vx(p.x - v.x, p.cs) : p + in_cs(v, p.cs)
(-)(p::Union{VX,VXY}, v::Union{VX,VXY,VPol,VPold}) =
    p.cs === v.cs ? vxy(p.x - v.x, p.y - v.y, p.cs) : p + in_cs(v, p.cs)
(-)(p::Union{VX,VXY,VXYZ}, v::Union{VX,VXY,VXYZ,VPol,VPold,VXYZ,VCyl,VSph}) =
    p.cs === v.cs ? vxyz(p.x - v.x, p.y - v.y, p.z - v.z, p.cs) : p + in_cs(v, p.cs)
(-)(p::VPol, v::Union{VX,VXY,VPol,VPold}) =
    vpol(vxy(p) - v)
(-)(p::VCyl, v::Union{VX,VXY,VXYZ,VPol,VPold,VCyl,VSph}) =
    vcyl(vxyz(p) - v)
(-)(p::VSph, v::Union{VX,VXY,VXYZ,VPol,VPold,VCyl,VSph}) =
    vsph(vxyz(p) - v)

(/)(p::VX, r::Real) =
    vx(p.x/r, p.cs)
(/)(p::VXY, r::Real) =
    vxy(p.x/r, p.y/r, p.cs)
(/)(p::VXYZ, r::Real) =
    vxyz(p.x/r, p.y/r, p.z/r, p.cs)
(/)(p::VPol, r::Real) =
    vpol(p.ρ/r, p.ϕ, p.cs)
(/)(p::VCyl, r::Real) =
    vcyl(p.ρ/r, p.ϕ, p.z, p.cs)
(/)(p::VSph, r::Real) =
    vsph(p.ρ/r, p.ϕ, p.ψ, p.cs)

    #=
(-)(p::X, v::VX) = p.cs === v.cs ? X(p.x - v.x, p.cs) : p + in_cs(c, p.cs)
(-)(p::XY, v::VXY) = p.cs === v.cs ? XY(p.x - v.x, p.y - v.y, p.cs) : p + in_cs(c, p.cs)

(-)(p::X, v::VXY) = p.cs === v.cs ? XY(p.x - v.x, p.y - v.y, p.cs) : p + in_cs(c, p.cs)

(+)(a::XYZ, b::VXYZ) = xyz(a.raw + in_cs(b, a.cs).raw, a.cs)
(+)(a::VXYZ, b::XYZ) = xyz(a.raw + in_cs(b, a.cs).raw, a.cs)
(+)(a::VXYZ, b::VXYZ) = vxyz(a.raw + in_cs(b, a.cs).raw, a.cs)
(-)(a::XYZ, b::VXYZ) = xyz(a.raw - in_cs(b, a.cs).raw, a.cs)
(-)(a::VXYZ, b::VXYZ) = vxyz(a.raw - in_cs(b, a.cs).raw, a.cs)
(-)(a::XYZ, b::XYZ) = vxyz(a.raw - in_cs(b, a.cs).raw, a.cs)
(-)(a::VXYZ) = vxyz(-a.raw, a.cs)
(*)(a::VXYZ, b::Real) = vxyz(a.raw * b, a.cs)
(*)(a::Real, b::VXYZ) = vxyz(a * b.raw, b.cs)
(/)(a::VXYZ, b::Real) = vxyz(a.raw / b, a.cs)
=#

# TO BE PROCESSED (and possibly removed)
xyz(s::Vec4f,cs::CS) =
  XYZ(s[1], s[2], s[3], cs, s)
#
vxyz(s::Vec4f,cs::CS) =
  VXYZ(s[1], s[2], s[3], cs, s)

# TODO add other fields, e.g.,
# Base.getproperty(p::XYZ, f::Symbol) = f === :rho ? pol_rho(p) : ...

zero(::Type{<:Loc}) = u0()

# Basic conversions
# From tuples of Loc
convert(::Type{Locs}, ps::NTuple{N,Loc}) where {N} = collect(XYZ, ps)

# From arrays of Any. This looks like a failure in Julia type inference, particularly when
# an empty array is involved, e.g., line(vcat([xy(10,20), xy(30,40)], []))
convert(::Type{Locs}, ps::Vector{<:Any}) = collect(Loc, ps)

scaled_cs(p::XYZ, x::Real, y::Real, z::Real) = xyz(p.x, p.y, p.z, scaled_cs(p.cs, x, y, z))
center_scaled_cs(p::XYZ, x::Real, y::Real, z::Real) = xyz(p.x/x, p.y/y, p.z/z, center_scaled_cs(p.cs, x, y, z))


const min_norm = 1e-20

unitized(v::Vec) =
  let r = sqrt(sum(abs2, v.raw))
    @assert r > min_norm "The vector $(v) is too small (norm: $(r)) to be unitized."
    vxyz(v.raw./r, v.cs)
  end

#=
There are two important operations with coordinate systems:
 - in_cs: converts a location or vector from one coordinate system to another, preserving its absolute location or orientation.
 - on_cs: transports a location or vector from one coordinate system to another, changing its absolute location or orientation. 
=#

in_cs(from_cs::CS, to_cs::CS) =
    to_cs === world_cs ?
        from_cs.transform :
        from_cs === world_cs ?
          inv(to_cs.transform) :
          inv(to_cs.transform) * from_cs.transform

in_cs(p::Loc, cs::CS) =
  p.cs === cs ?
    p :
    cs === world_cs ?
      xyz(p.cs.transform * p.raw, world_cs) :
      xyz(inv(cs.transform) * p.cs.transform * p.raw, cs)

in_cs(p::Vec, cs::CS) =
  p.cs === cs ?
    p :
    cs === world_cs ?
      vxyz(p.cs.transform * p.raw, world_cs) :
      vxyz(inv(cs.transform) * p.cs.transform * p.raw, cs)

in_cs(p, q) = in_cs(p, q.cs)

in_world(p) = in_cs(p, world_cs)


on_cs(p::Loc, cs::CS) =
  p.cs === cs ?
    p :
    xyz(raw_point(p)..., cs)

on_cs(ps::Locs, cs::CS) =
  [on_cs(p, cs) for p in ps]

on_cs(p, q::Loc) = on_cs(p, translated_cs(q.cs, q.x, q.y, q.z))



export inverse_transformation
inverse_transformation(cs::CS) = CS(inv(cs.transform))
inverse_transformation(p::Loc) = xyz(0,0,0, CS(inv(translated_cs(p.cs, p.x, p.y, p.z).transform)))

cs_from_o_vx_vy_vz(o::Loc, ux::Vec, uy::Vec, uz::Vec) =
  CS(SMatrix{4,4,Float64}(ux.x, ux.y, ux.z, 0, uy.x, uy.y, uy.z, 0, uz.x, uz.y, uz.z, 0, o.x, o.y, o.z, 1))

LinearAlgebra.cross(v::Vec, w::Vec) = _cross(v.raw, in_cs(w, v.cs).raw, v.cs)
_cross(a::Vec4f, b::Vec4f, cs::CS) =
  vxyz(a[2]*b[3]-a[3]*b[2], a[3]*b[1]-a[1]*b[3], a[1]*b[2]-a[2]*b[1], cs)

LinearAlgebra.dot(v::Vec, w::Vec) = _dot(v.raw, in_cs(w, v.cs).raw)
_dot(a::Vec4f, b::Vec4f) =
  a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

cs_from_o_vx_vy(o::Loc, vx::Vec, vy::Vec) =
  let o = in_world(o),
    vx = unitized(in_world(vx)),
    vz = unitized(cross(vx, in_world(vy)))
    cs_from_o_vx_vy_vz(o, vx, cross(vz,vx), vz)
  end

cs_from_o_vz(o::Loc, n::Vec) =
  let o = in_world(o),
      n = in_world(n),
      vx = vpol(1, sph_phi(n) + pi/2, o.cs),
      vy = unitized(cross(n, vx)),
      vz = unitized(n)
    cs_from_o_vx_vy_vz(o, vx, vy, vz)
  end
#=
cs_from_o_phi(o::Loc, phi::Real) =
  let vx = in_world(vcyl(1, phi, 0, o.cs))
      vy = in_world(vcyl(1, phi + pi/2, 0, o.cs))
      vz = cross(vx, vy)
      o = in_world(o)
      cs_from_o_vx_vy_vz(o, vx, vy, vz)
  end
  =#
cs_from_o_phi(o::Loc, ϕ::Real) =
  rotated_z_cs(translated_cs(o.cs, o.x, o.y, o.z), ϕ)
cs_from_o_rot_x(o::Loc, ϕ::Real) =
  rotated_x_cs(translated_cs(o.cs, o.x, o.y, o.z), ϕ)
cs_from_o_rot_y(o::Loc, ϕ::Real) =
  rotated_y_cs(translated_cs(o.cs, o.x, o.y, o.z), ϕ)
cs_from_o_rot_z(o::Loc, ϕ::Real) =
  rotated_z_cs(translated_cs(o.cs, o.x, o.y, o.z), ϕ)
cs_from_o_rot_zyx(o::Loc, z::Real, y::Real, x::Real) =
  rotated_x_cs(rotated_y_cs(rotated_z_cs(translated_cs(o.cs, o.x, o.y, o.z), z), y), x)

loc_from_o_vx(o::Loc, vx::Vec) = loc_from_o_vx_vy(o, vx, vpol(1, pol_phi(vx) + pi/2))
loc_from_o_vx_vy(o::Loc, vx::Vec, vy::Vec) = u0(cs_from_o_vx_vy(o, vx, vy))
loc_from_o_vz(o::Loc, vz::Vec) = u0(cs_from_o_vz(o, vz))
loc_from_o_phi(o::Loc, ϕ::Real) = u0(cs_from_o_phi(o, ϕ))
loc_from_o_rot_x(o::Loc, ϕ::Real) = u0(cs_from_o_rot_x(o, ϕ))
loc_from_o_rot_y(o::Loc, ϕ::Real) = u0(cs_from_o_rot_y(o, ϕ))
loc_from_o_rot_z(o::Loc, ϕ::Real) = u0(cs_from_o_rot_z(o, ϕ))
loc_from_o_rot_zyx(o::Loc, z::Real, y::Real, x::Real) = u0(cs_from_o_rot_zyx(o, z, y, x))

#To handle the common case
maybe_loc_from_o_vz(o::Loc, n::Vec) =
  let n = in_world(n)
    if n.x == 0 && n.y == 0
      o
    else
      loc_from_o_vz(o, n)
    end
  end

#This is not needed!
#(+){T1,T2,T3,T4,T5,T6}(p::XYZ{T1,T2,T3},v::VXYZ{T4,T5,T6}) = xyz(p.x+v.x, p.y+v.y, p.z+v.z, p.raw+v.raw)


add_x(p::Loc, x::Real) = xyz(p.x+x, p.y, p.z, p.cs)
add_y(p::Loc, y::Real) = xyz(p.x, p.y+y, p.z, p.cs)
add_z(p::Loc, z::Real) = xyz(p.x, p.y, p.z+z, p.cs)
add_xy(p::Loc, x::Real, y::Real) = xyz(p.x+x, p.y+y, p.z, p.cs)
add_xz(p::Loc, x::Real, z::Real) = xyz(p.x+x, p.y, p.z+z, p.cs)
add_yz(p::Loc, y::Real, z::Real) = xyz(p.x, p.y+y, p.z+z, p.cs)
add_xyz(p::Loc, x::Real, y::Real, z::Real) = xyz(p.x+x, p.y+y, p.z+z, p.cs)
# FIX THIS
add_pol(p::Loc, ρ::Real, ϕ::Real) = p + vcyl(ρ, ϕ, 0, p.cs)
add_cyl(p::Loc, ρ::Real, ϕ::Real, z::Real) = p + vcyl(ρ, ϕ, z, p.cs)
add_vcyl(v::Vec, ρ::Real, ϕ::Real, z::Real) = v + vcyl(ρ, ϕ, z, v.cs)
add_vpol(v::Vec, ρ::Real, ϕ::Real) = add_vcyl(v, ρ, ϕ, 0)
add_vsph(v::Vec, ρ::Real, ϕ::Real, ψ::Real) = v + vsph(ρ, ϕ, ψ, v.cs)
add_sph(p::Loc, ρ::Real, ϕ::Real, ψ::Real) = p + vsph(ρ, ϕ, ψ, p.cs)



norm(v::Vec) = norm(v.raw)
length(v::Vec) = norm(v.raw)

min_loc(p::Loc, q::Loc) =
    xyz(min.(p.raw, in_cs(q, p.cs).raw), p.cs)
max_loc(p::Loc, q::Loc) =
    xyz(max.(p.raw, in_cs(q, p.cs).raw), p.cs)

distance(p::Loc, q::Loc) = norm((in_world(q)-in_world(p)).raw)

u0(cs=current_cs())   = x(0,cs)
ux(cs=current_cs())   = xyz(1,0,0,cs)
uy(cs=current_cs())   = xyz(0,1,0,cs)
uz(cs=current_cs())   = xyz(0,0,1,cs)
uxy(cs=current_cs())  = xy(1,1,cs)
uyz(cs=current_cs())  = yz(1,1,cs)
uxz(cs=current_cs())  = xz(1,1,cs)
uxyz(cs=current_cs()) = xyz(1,1,1,cs)

#=
x(x::Real=1,cs=current_cs()) = xyz(x,0,0,cs)
y(y::Real=1,cs=current_cs()) = xyz(0,y,0,cs)
z(z::Real=1,cs=current_cs()) = xyz(0,0,z,cs)
xy(x::Real=1,y::Real=1,cs=current_cs()) = xyz(x,y,0,cs)
yz(y::Real=1,z::Real=1,cs=current_cs()) = xyz(0,y,z,cs)
xz(x::Real=1,z::Real=1,cs=current_cs()) = xyz(x,0,z,cs)
=#
uvx(cs=current_cs())   = vxyz(1,0,0,cs)
uvy(cs=current_cs())   = vxyz(0,1,0,cs)
uvz(cs=current_cs())   = vxyz(0,0,1,cs)
uvxy(cs=current_cs())  = vxyz(1,1,0,cs)
uvyz(cs=current_cs())  = vxyz(0,1,1,cs)
uvxz(cs=current_cs())  = vxyz(1,0,1,cs)
uvxyz(cs=current_cs()) = vxyz(1,1,1,cs)

position_and_height(p, q) = loc_from_o_vz(p, q - p), distance(p, q)

regular_polygon_vertices(edges::Integer=3, center::Loc=u0(), radius::Real=1, angle::Real=0, is_inscribed::Bool=true) = begin
  r = is_inscribed ? radius : radius/cos(pi/edges)
  [center + vpol(r, a, center.cs) for a in division(angle, angle + 2*pi, edges, false)]
end

intermediate_loc(p::Loc, q::Loc, f::Real=0.5) =
  if p.cs == q.cs
    p+(q-p)*f
  else
    o = intermediate_loc(in_world(p), in_world(q), f)
    v_x = in_world(vx(1, p.cs))*(1-f) + in_world(vx(1, q.cs))*f
    v_y = in_world(vy(1, p.cs))*(1-f) + in_world(vy(1, q.cs))*f
    loc_from_o_vx_vy(o, v_x, v_y)
  end

# Metaprogramming

meta_program(x::Any) = x # literals might be self evaluating
meta_program(x::Int) = x
meta_program(x::Real) = round(x,sigdigits=8)
meta_program(x::Bool) = x
meta_program(x::DataType) = Symbol(x)
meta_program(x::Vector{T}) where T = 
  isempty(x) ?
    Expr(:ref, meta_program(T)) :
    Expr(:vect, map(meta_program, x)...)
meta_program(p::Loc) =
    if cz(p) == 0
        Expr(:call, :xy, meta_program(cx(p)), meta_program(cy(p)))
    else
        Expr(:call, :xyz, meta_program(cx(p)), meta_program(cy(p)), meta_program(cz(p)))
    end
meta_program(v::Vec) =
    if cz(v) == 0
        Expr(:call, :vxy, meta_program(cx(v)), meta_program(cy(v)))
    else
        Expr(:call, :vxyz, meta_program(cx(v)), meta_program(cy(v)), meta_program(cz(v)))
    end

# Conversions
# We could accept some nice conversions
# convert(::Type{Loc}, t::Tuple{Real,Real,Real}) = xyz(t[1], t[2], t[3])


# Integration in standard protocols

# iteration for destructuring into components
iterate(v::Vec) = iterate(v.raw)
iterate(v::Vec, state) = iterate(v.raw, state)

iterate(v::Loc) = iterate(v.raw)
iterate(v::Loc, state) = iterate(v.raw, state)

# Utilities
export trig_center, trig_normal, quad_center, quad_normal, vertices_center, vertices_normal, iterate_quads

trig_center(p0, p1, p2) =
  xyz((p0.x+p1.x+p2.x)/3, (p0.y+p1.y+p2.y)/3, (p0.z+p1.z+p2.z)/3, p0.cs)

trig_normal(p0, p1, p2) =
  polygon_normal([p1 - p0, p2 - p1, p0 - p2])

quad_center(p0, p1, p2, p3) =
  intermediate_loc(intermediate_loc(p0, p2), intermediate_loc(p1, p3))

quad_normal(p0, p1, p2, p3) =
  let p0 = in_world(p0),
      p1 = in_world(p1),
      p2 = in_world(p2),
      p3 = in_world(p3)
    polygon_normal([p1 - p0, p2 - p1, p3 - p2, p0 - p3])
  end

vertices_center(pts) =
  let pts = map(in_world, pts),
      n=length(pts),
      xs=[cx(p) for p in pts],
      ys=[cy(p) for p in pts],
      zs=[cz(p) for p in pts]
    xyz(sum(xs)/n, sum(ys)/n, sum(zs)/n, world_cs)
  end

vertices_normal(ps) =
  let ps = map(in_world, ps)
    polygon_normal(p-q for (p,q) in zip(ps, drop(cycle(ps), 1)))
  end

polygon_normal(vs) =
  unitized(
    sum(
      cross(v0,v1)
      for (v0,v1) in zip(vs, drop(cycle(vs), 1))))

iterate_quads(f, ptss) =
  [[f(p0, p1, p2, p3)
    for (p0, p1, p2, p3)
    in zip(pts0[1:end-1], pts1[1:end-1], pts1[2:end], pts0[2:end])]
    for (pts0, pts1)
    in zip(ptss[1:end-1], ptss[2:end])]


# We need to implement smooth walks along curves
# Using just the Frenet frame is not adequate as
# changes in curvature cause it to suddenly change
# direction.

# The technique we use to solve this is based on
# rotation minimizing frames (RMF) presented in
# "Computation of Rotation Minimizing Frames"
# (Wenping Wang, Bert Jüttler, Dayue Zheng, and Yang Liu, 2008)
# Regarding the paper notation, t = vz, r = -vx, s = vy
export rotation_minimizing_frames

rotation_minimizing_frames(frames) =
  rotation_minimizing_frames(
    frames[1],
    [in_world(frame) for frame in frames],
    [in_world(vz(1, frame.cs)) for frame in frames])

#=
  let new_frames = [frames[1]]
    for x1 in drop(frames, 1)
      # Reflection of x0 tangent and axis onto x1
      # using reflection plane located between x0 and x1
      let x0 = new_frames[end],
          v1 = in_world(x1) - in_world(x0),
          c1 = dot(v1, v1),
          r0 = in_world(vx(1, x0.cs)),
          t0 = in_world(vz(1, x0.cs)),
          ril = r0 - v1*(2/c1*dot(v1,r0)),
          til = t0 - v1*(2/c1*dot(v1,t0)),
          # Reflection on a plane at x1, aligning the frame
          # tangent with the curve tangent
          t1 = in_world(vz(1, x1.cs)),
          v2 = t1 - til,
          c2 = dot(v2, v2),
          r1 = ril - v2*(2/c2*dot(v2, ril)),
          s1 = cross(t1, r1)
        push!(new_frames, loc_from_o_vx_vy(x1, r1, s1))
      end
    end
    new_frames
  end
=#

rotation_minimizing_frames(u0, xs, ts) =
  let ri = in_world(vy(1, u0.cs)),
      new_frames = [loc_from_o_vx_vy(xs[1], cross(ri, ts[1]), ri)]
    for i in 1:length(xs)-1
      let xi = xs[i],
          xii = xs[i+1],
          ti = ts[i],
          tii = ts[i+1],
          v1 = xii - xi,
          c1 = dot(v1, v1),
          ril = ri - v1*(2/c1*dot(v1,ri)),
          til = ti - v1*(2/c1*dot(v1,ti)),
          v2 = tii - til,
          c2 = dot(v2, v2),
          rii = ril - v2*(2/c2*dot(v2, ril)),
          sii = cross(rii, tii),
          uii = loc_from_o_vx_vy(xii, sii, rii)
        push!(new_frames, uii)
        ri = rii
      end
    end
    new_frames
  end

#=
TODO: Consider adding vectors without frame of reference:

Δx(x) = vx(x)

p + Δx(5)
p + Δxy(1,2)
Addition would respect the frame of reference of p

Another hypotesis is to write

p + _x(5)
p + _xy(1,2)

or even

p + dx(5)
p + dxy(1,2)

For the moment, I'll choose this one
=#


# Broadcasting

broadcastable(p::Loc) = Ref(p)
broadcastable(v::Vec) = Ref(v)

# equality

Base.isequal(p::Loc, q::Loc) =
  let wp = in_world(p),
      wq = in_world(q)
    isequal(wp.x, wq.x) && isequal(wp.y, wq.y) && isequal(wp.z, wq.z)
  end

Base.isapprox(p::Loc, q::Loc; kwargs...) =
  let wp = in_world(p),
      wq = in_world(q)
      isapprox(wp.x, wq.x; kwargs...) && isapprox(wp.y, wq.y; kwargs...) && isapprox(wp.z, wq.z; kwargs...)
  end

Base.isequal(v::Vec, w::Vec) =
  let wv = in_world(v),
      ww = in_world(w)
    isequal(wv.x, ww.x) && isequal(wv.y, ww.y) && isequal(wv.z, ww.z)
  end

# angle between

angle_between(v1, v2) =
  let v1 = in_world(v1),
      v2 = in_world(v2)
    acos(dot(v1, v2)/(norm(v1)*norm(v2)))
  end

perpendicular_point(p, n, q) =
  let n = unitized(n)
    p + dot((q - p), n)*n
  end

################################################################################
# To embed Khepri, it becomes useful to convert entities into 'raw' data

raw_point(v::Union{Loc, Vec}) =
  let o = in_world(v)
    (float(o.x), float(o.y), float(o.z))
  end

raw_plane(v::Loc) =
  let o = in_world(v),
      vx = in_world(vx(1, v.cs))
      vy = in_world(vy(1, v.cs))
    (float(o.x), float(o.y), float(o.z),
     float(vx.x), float(vx.y), float(vx.z),
     float(vy.x), float(vy.y), float(vy.z))
  end

################################################################################
export acw_vertices
acw_vertices(vs) =
  angle_between(vertices_normal(vs), vz()) < pi/4 ?
    vs :
    reverse(vs)


###
# We need vectors of uniform coordinate dimensions

Base.convert(::Type{Vector{Loc2D}}, ps::Vector{<:Loc3D}) =
  isempty(ps) ?
    Loc2D[] :
    let cs = cs_from_o_vz(vertices_center(ps), vertices_normal(ps))
      [convert(Loc2D, in_cs(p, cs)) for p in ps]
    end

Base.convert(::Type{Loc2D}, p::XYZ) =
  p.z ≈ 0.0 ?
    xy(p.x, p.y, p.cs) :
    error("Can't convert to a Loc2D the non-zero z location $p")


