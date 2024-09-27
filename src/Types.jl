#=

Here we define the relevant types used in Khepri.

=#


export Loc1D, Loc2D, Loc3D, Loc, 
       Vec1D, Vec2D, Vec3D, Vec,
       Locs1D, Locs2D, Locs3D, Locs,
       Vecs1D, Vecs1D, Vecs1D, Vecs

# We distinguish between locations and vectors, as they transform differently

abstract type Loc end
abstract type Vec end

# We distinguish between 1D, 2D, and 3D coords.
# Note that any place that accepts a 3D location can also accept a 2D location or a 1D location.
# Similarly, any place that accepts a 2D location can also accept a 1D location.
# The opposite is not true.
# This means that we have the subtype hierarchy 1D <: 2D <: 3D

abstract type Loc3D <: Loc end
abstract type Loc2D <: Loc3D end
abstract type Loc1D <: Loc2D end

abstract type Vec3D <: Vec end
abstract type Vec2D <: Vec3D end
abstract type Vec1D <: Vec2D end

# We need vectors of Locs and vectors of Vecs

const Locs = Vector{<:Loc}
const Vecs = Vector{<:Vec}

# Special cases are also useful

const Locs1D = Vector{<:Loc1D}
const Locs2D = Vector{<:Loc2D}
const Locs3D = Vector{<:Loc3D}

const Vecs1D = Vector{<:Vec1D}
const Vecs2D = Vector{<:Vec2D}
const Vecs3D = Vector{<:Vec3D}

#=
Paths and Regions are abstract entities that describe geometric curves
and regions.
They belong to a hierarchy of geometry elements
=#

abstract type GeometryElement end
abstract type Path <: GeometryElement end
abstract type OpenPath <: Path end
abstract type ClosedPath <: Path end

#=
Regions are areas delimited by paths. They are assumed to be planar and
might contain holes.
=#

struct Region
  paths::Vector{<:ClosedPath}
end


#=
To represent GeometryElements, we need to create graphical entities in a
given backend. Sometimes, we need to create non-graphical entities, such
as layers or materials. In both cases, we need proxies.
=#

abstract type Proxy end

#=
Shapes are a particular type of proxy:
=#

abstract type Shape{D} <: Proxy end

#=
Shapes can be specialized according to their dimensions:
=#

const Shape0D = Shape{0}
const Shape1D = Shape{1}
const Shape2D = Shape{2}
const Shape3D = Shape{3}

#=
Common predicates can be defined for these subtypes
=#

is_curve(s::Shape) = false
is_surface(s::Shape) = false
is_solid(s::Shape) = false

is_curve(s::Shape1D) = true
is_surface(s::Shape2D) = true
is_solid(s::Shape3D) = true

#=
Vectors of shapes are also useful:
=#

const Shapes = Vector{<:Shape}

# HACK: Fix element type
const Shapes0D = Vector{<:Any}
const Shapes1D = Vector{<:Any}
const Shapes2D = Vector{<:Any}
