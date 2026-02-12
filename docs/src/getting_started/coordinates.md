# Coordinates and Vectors

KhepriBase distinguishes between **locations** (points in space) and
**vectors** (displacements/directions). A translation applied to a location
moves it; the same translation leaves a direction vector unchanged.

Both are represented internally as 4-component homogeneous coordinates
(`SVector{4,Float64}`). Locations have a fourth component of `1.0`; vectors
have `0.0`, letting all affine transformations be a single 4x4 matrix multiply.

Source: `src/Coords.jl`

## Type Hierarchy

```
Loc                Vec
 +-- Loc3D          +-- Vec3D
      +-- Loc2D          +-- Vec2D
           +-- Loc1D          +-- Vec1D
```

A 1D location can be used where a 2D or 3D one is expected. Concrete types:

| Struct | Supertype | Coords | Fields | Vector variant |
|--------|-----------|--------|--------|----------------|
| `X`    | `Loc1D`   | Cartesian 1D    | `x`             | `VX`    |
| `XY`   | `Loc2D`   | Cartesian 2D    | `x, y`          | `VXY`   |
| `XYZ`  | `Loc3D`   | Cartesian 3D    | `x, y, z`       | `VXYZ`  |
| `Pol`  | `Loc2D`   | Polar (rad)     | `rho, phi`       | `VPol`  |
| `Pold` | `Loc2D`   | Polar (deg)     | `rho, phi`       | `VPold` |
| `Cyl`  | `Loc3D`   | Cylindrical     | `rho, phi, z`    | `VCyl`  |
| `Sph`  | `Loc3D`   | Spherical       | `rho, phi, psi`  | `VSph`  |

Every instance stores its native fields, a `CS` reference, and a precomputed
`raw::Vec4f` with the Cartesian representation.

## Creating Locations

```julia
x(3)              # (3, 0, 0)
y(5)              # (0, 5, 0)
z(2)              # (0, 0, 2)
xy(3, 5)          # (3, 5, 0)
xz(3, 2)          # (3, 0, 2)
yz(5, 2)          # (0, 5, 2)
xyz(3, 5, 2)      # (3, 5, 2)
pol(5, pi/4)      # polar (radians)
pold(5, 45)       # polar (degrees)
cyl(5, pi/4, 10)  # cylindrical
sph(5, pi/4, pi/6)# spherical
```

All constructors accept an optional trailing `cs` argument (defaults to
`current_cs()`). Location parameters default to `0`.

## Creating Vectors

Vector constructors mirror locations with a `v` prefix. Parameters default
to `1` (not `0`).

```julia
vx(3)               vy(5)               vz(2)
vxy(3, 5)           vxyz(3, 5, 2)
vpol(5, pi/4)       vcyl(5, pi/4, 10)   vsph(5, pi/4, pi/6)
```

## Unit Locations and Vectors

| Location  | Value     | Vector   | Value     |
|-----------|-----------|----------|-----------|
| `u0()`    | (0,0,0)   | `uvx()`  | (1,0,0)   |
| `ux()`    | (1,0,0)   | `uvy()`  | (0,1,0)   |
| `uy()`    | (0,1,0)   | `uvz()`  | (0,0,1)   |
| `uz()`    | (0,0,1)   | `uvxy()` | (1,1,0)   |
| `uxy()`   | (1,1,0)   | `uvyz()` | (0,1,1)   |
| `uyz()`   | (0,1,1)   | `uvxz()` | (1,0,1)   |
| `uxz()`   | (1,0,1)   | `uvxyz()`| (1,1,1)   |
| `uxyz()`  | (1,1,1)   |          |           |

All accept an optional `cs` argument.

## Component Accessors

Cartesian: `cx(p)`, `cy(p)`, `cz(p)` -- work on any `Loc` or `Vec`.
Polar: `pol_rho(p)`, `pol_phi(p)`.
Cylindrical: `cyl_rho(p)`, `cyl_phi(p)`, `cyl_z(p)`.
Spherical: `sph_rho(p)`, `sph_phi(p)`, `sph_psi(p)`.

Property access also works across coordinate types:

```julia
p = pol(5, pi/4)
p.x   # 5*cos(pi/4)
p.y   # 5*sin(pi/4)
p.z   # 0
```

## Arithmetic

The type system enforces correct geometric semantics:

| Operation | Result | Meaning |
|-----------|--------|---------|
| `loc + vec` | `Loc` | Displace a point |
| `vec + loc` | `Loc` | Same (commutative) |
| `loc - loc` | `Vec` | Displacement between points |
| `loc - vec` | `Loc` | Displace backwards |
| `vec + vec` | `Vec` | Combine displacements |
| `vec - vec` | `Vec` | Difference of displacements |
| `vec * scalar` | `Vec` | Scale |
| `scalar * vec` | `Vec` | Scale |
| `vec / scalar` | `Vec` | Scale |
| `-vec` | `Vec` | Reverse direction |

```julia
p = xyz(1, 2, 3)
v = vxyz(10, 0, 0)
p + v        # xyz(11, 2, 3)
p + v - p    # vxyz(10, 0, 0)
v * 2        # vxyz(20, 0, 0)
```

When operands belong to different coordinate spaces, the right operand is
automatically converted into the left operand's CS via `in_cs`.

### Geometric functions

```julia
distance(p, q)         # Euclidean distance between two locations
norm(v)                # length of a vector
unitized(v)            # unit vector (error if near-zero)
dot(v, w)              # scalar dot product
cross(v, w)            # cross product (returns Vec)
angle_between(v1, v2)  # angle in radians
```

## Displacement Functions

Convenience functions that offset a location along specific axes within its
own coordinate space:

```julia
add_x(p, dx)                add_y(p, dy)              add_z(p, dz)
add_xy(p, dx, dy)           add_xz(p, dx, dz)         add_yz(p, dy, dz)
add_xyz(p, dx, dy, dz)
add_pol(p, rho, phi)        add_cyl(p, rho, phi, z)    add_sph(p, rho, phi, psi)
```

## Coordinate Spaces

A `CS` is a 4x4 affine transformation matrix defining an origin and three
axis directions relative to the world frame.

### World and current CS

```julia
world_cs       # global constant identity CS
current_cs     # a Parameter{CS}, defaults to world_cs
current_cs()   # read current value
```

### Converting between coordinate spaces

```julia
in_cs(p, cs)   # express p in cs (preserves absolute position)
in_world(p)    # shorthand for in_cs(p, world_cs)
on_cs(p, cs)   # transport p onto cs (keeps local coords, changes absolute position)
```

- `in_cs` answers "what are the coordinates of this same point in another frame?"
- `on_cs` answers "place this local-coordinate point into another frame."

### Temporarily changing the current CS

Use `translating_current_cs` to shift `current_cs` within a scoped block:

```julia
translating_current_cs(dx=10) do
  sphere(u0(), 1)   # centered at the translated origin
end
```

### Building coordinate spaces

```julia
cs_from_o_vx_vy(o, vx, vy)        # origin + x-axis + y-axis (z = cross product)
cs_from_o_vx_vy_vz(o, vx, vy, vz) # origin + all three axes
cs_from_o_vz(o, n)                 # origin + z-axis (normal)
cs_from_o_phi(o, phi)              # origin + rotation around z
```

From an origin and rotation angles:

```julia
cs_from_o_rot_x(o, phi)            # rotate around x-axis
cs_from_o_rot_y(o, phi)            # rotate around y-axis
cs_from_o_rot_z(o, phi)            # rotate around z-axis
cs_from_o_rot_zyx(o, z, y, x)     # Euler angles: z, then y, then x
```

Low-level CS transformations:

```julia
translated_cs(cs, x, y, z)        # translate
scaled_cs(cs, x, y, z)            # scale along axes
rotated_x_cs(cs, alpha)           # rotate around x
rotated_y_cs(cs, alpha)           # rotate around y
rotated_z_cs(cs, alpha)           # rotate around z
```

### Location-returning variants

Each `cs_from_*` has a `loc_from_*` counterpart that returns `u0()` at the
origin of the new CS: `loc_from_o_vx`, `loc_from_o_vx_vy`, `loc_from_o_vz`,
`loc_from_o_phi`, `loc_from_o_rot_x`, `loc_from_o_rot_y`, `loc_from_o_rot_z`,
`loc_from_o_rot_zyx`. These are commonly used to attach a local frame to a
point along a [path](paths.md).

## Coordinate Conversion

Any location or vector can be converted to a different representation:

```julia
p = pol(5, pi/4)
xyz(p)    # to Cartesian
cyl(p)    # to Cylindrical

q = xyz(3, 4, 5)
sph(q)    # to Spherical
pol(q)    # to Polar (projects to xy-plane)
```

For vectors: `vxy(v)`, `vpol(v)`, `vcyl(v)`, `vsph(v)`.

## Utilities

| Function | Description |
|----------|-------------|
| `intermediate_loc(p, q, f=0.5)` | Interpolate position (and frame) between two locs |
| `min_loc(p, q)` / `max_loc(p, q)` | Component-wise min/max |
| `regular_polygon_vertices(n, center, r)` | Vertices of a regular polygon |
| `rotation_minimizing_frames(frames)` | Smooth frames along a curve (Wang et al. 2008) |

The last function avoids Frenet frame discontinuities and is used internally
by [shape](../concepts/shapes.md) sweep operations along [paths](paths.md).

## Equality, Broadcasting, and Destructuring

Locations and vectors are compared by world-space coordinates:

```julia
isequal(xyz(1, 0, 0), pol(1, 0))   # true
isapprox(p, q)                      # approximate comparison
```

They opt out of broadcasting (act as scalars) and support iteration for
destructuring into four homogeneous components:

```julia
(a, b, c, d) = vxyz(1, 2, 3)   # a=1.0, b=2.0, c=3.0, d=0.0
```