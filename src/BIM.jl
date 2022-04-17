#####################################################################
# BIM
#=
Building Information Modeling is more than just shapes.
Each BIM element is connected to other BIM elements. A wall is connected to its
windows and doors. A floor is connected to its walls, and stairs. Stairs connect
different floors, etc.

This graph of objects needs to be build programmatically and that is one problem.
The other problem is portability, as we want this to work in backends that are
not BIM tools.

A third problem is related to level of detail. Even in the same backend, we might
need to represent BIM elements at different levels of detail.

To solve these problems, we will use a protocol to realize BIM objects. To avoid
confusion, we need to ensure a series of invariants, namely, that entities are
portable above a certain level and non-portable below that level. As an example,
a Slab and a SlabFamily are portable entities.

We will start with the simplest of BIM elements, namely, levels, slabs, walls,
windows and doors.
=#

abstract type BIMElement <: Proxy end
abstract type Measure <: BIMElement end
BIMElements = Vector{<:BIMElement}

# Level
@defproxy(level, Measure, height::Real=0, elements::BIMElements=BIMElement[])
levels_cache = Dict{Real,Level}()
maybe_replace(level::Level) = get!(levels_cache, level.height, level)

convert(::Type{Level}, h::Real) = level(h)

current_levels() = values(level_cache)
default_level = OptionParameter{Level}(level())
default_level_to_level_height = Parameter{Real}(3)
upper_level(lvl=default_level(), height=default_level_to_level_height()) = level(lvl.height + height)
Base.:(==)(l1::Level, l2::Level) = l1.height == l2.height

#default implementation
realize(b::Backend, s::Level) = s.height

export all_levels, default_level, default_level_to_level_height, upper_level

#=
@defproxy(polygonal_mass, BIMShape, points::Locs, height::Real)
@defproxy(rectangular_mass, BIMShape, center::Loc, width::Real, len::Real, height::Real)

@defproxy(column, BIMShape, center::Loc, bottom_level::Any, top_level::Any, family::Any)
=#

abstract type BIMShape <: Shape3D end

#=

We need to provide defaults for a lot of things. For example, we want to specify
a wall that goes through a path without having to specify the kind of wall or its
thickness and height.

This means that, apart from the wall's path, all other wall features will come
from defaults. The base height will be determined by the current level and the
wall height by the current level-to-level height. Finally, the wall thickness,
constituent parts, thermal characteristics, and so on will come from the wall
defaults.  In the case of wall, we will assume that current_wall_defaults() is
a parameter that contains a set of wall parameters.  As an example of use, we
might have:

current_wall_defaults(wall_defaults(thickness=10))

Another option is the definition of different defaults:

thick_wall_defaults = wall_defaults(thickness=10)
thin_wall_defaults = wall_defaults(thickness=5)

which then can be made current:

current_wall_defaults(thin_wall_defaults)

In most cases, the defaults are not just one value, but a bunch of them. For a
beam, we might have:

standard_beam = beam_defaults(width=10, height=20)
current_beam_defaults(standard_beam)

Another useful feature is the ability to adapt defaults. For example:

current_beam_defaults(beam_with(standard_beam, height=20))

Finally, defaults can be created for anything. For example, in a building, we
might want to define a bunch of parameters that are relevant. The syntax is as
follows:

@defaults(building,
    width::Real=20,
    length::Real=30,
    height::Real=50)

In order to access these defaults, we can use the following:

current_building_defaults().width

In some cases, defaults are supported by the backend itself. For example, in
Revit, a wall can be specified using a family. In order to realize the wall
defaults in the current backend, we need to map from the wall parameters to the
corresponding BIM family parameters. This mapping must be described in a
different structure.

For example, a beam element might have a section with a given width and height
but, in Revit, a beam element such as "...\\Structural Framing\\Wood\\M_Timber.rfa"
has, as parameters, the dimensions b and d.  This means that we need a map, such
as Dict(:width => "b", :height => "d")))). So, for a Revit family, we might use:

RevitFamily(
    "C:\\ProgramData\\Autodesk\\RVT 2017\\Libraries\\US Metric\\Structural Framing\\Wood\\M_Timber.rfa",
    Dict(:width => "b", :height => "d"))

To make things more interesting, some families might require instantiation on
different levels. For example, a circular window family in Revit needs to be
loaded using RVTLoadFamily, which requires the name of the family, then it needs
to be instantiated with RVTFamilyElement, which requires the radius of the window,
and finally needs to be inserted on the wall, using RVTInsertWindow, which requires
the opening angle. This might be different for different families, so we need a
flexible way of using the parameters. One hipotesis is to specify those different
moments as different dictionaries.

RevitFamily(
  "C:\\ProgramData\\Autodesk\\RVT 2019\\Libraries\\US Metric\\Windows\\CIRCULAR WINDOW.rfa",
  ("radius_window" => :radius),
  ("angle_window" => :opening_angle))

If no parameters are needed on a particular phase, we might use an empty dictionary
to describe that phase. Given the typical Revit families, defaults parameters seem
to be the best approach here. This means that the previous beam family might be
equivalent to

RevitFamily(
  "C:\\ProgramData\\Autodesk\\RVT 2017\\Libraries\\US Metric\\Structural Framing\\Wood\\M_Timber.rfa",
  ("b" => :width, "d" => :height),
  ())

However, the same beam might have a different mapping in a different backend.
This means that we need another mapping to support different backends. One
possibility is to use something similar to:

backend_family(
    revit => RevitFamily(
        "C:\\ProgramData\\Autodesk\\RVT 2017\\Libraries\\US Metric\\Structural Framing\\Wood\\M_Timber.rfa",
        ("b" => :width, "d" => :height)),
    archicad => ArchiCADFamily(
        "BeamElement",
        ("size_x" => :width, "size_y" => :height)),
    autocad => AutoCADFamily())

Then, we need an operation that instantiates a family. This can be done on two different
levels: (1) from a backend-specific family (e.g., RevitFamily), for example:

beam_family = RevitFamily(
    "C:\\ProgramData\\Autodesk\\RVT 2017\\Libraries\\US Metric\\Structural Framing\\Wood\\M_Timber.rfa",
    ("b" => :width, "d" => :height))

current_beam_defaults(beam_family_instance(beam_family, width=10, height=20)

or from a generic backend family, for example:

beam_family = backend_family(
    revit => RevitFamily(
        "C:\\ProgramData\\Autodesk\\RVT 2017\\Libraries\\US Metric\\Structural Framing\\Wood\\M_Timber.rfa",
        ("b" => :width, "d" => :height)),
    archicad => ArchiCADFamily(
        "BeamElement"
        ("size_x" => :width, "size_y" => :height)),
    autocad => AutoCADFamily())

current_beam_defaults(beam_family_instance(beam_family, width=10, height=20)

In this last case, the generic family will use the top_backend value to identify
which family to use.

Another important feature is the use of a delegation-based implementation for
family instances. This means that we might do

current_beam_defaults(beam_family_instance(current_beam_defaults(), width=20)

to instantiate a family that uses, by default, the same parameter values used by
another family instance.

Finally, because we might need to know which families are available (e.g., for
metaprogramming purposes), we need a database of families for each BIM element.

A relatively simple approach is to have a registry for families:

family_registry(WallFamily) -> Vector{WallFamily}
register_family(family)
=#

export Family, FamilyInstance, family, family_ref

abstract type Family <: Proxy end
abstract type FamilyInstance <: Family end

family(f::Family) = f
family(f::FamilyInstance) = f.family

#=
Caching of families is more complex than caching of shapes because families
are created independently of the current backend and then they are lazily
instantiated, possibly multiples times as the current backend is being changed.
=#

family_ref(s::BIMShape) =
   family_ref(backend(s), s.family)
family_ref(b::Backend, f::Family) =
  get!(f.ref, b) do
    realize(b, f)
  end

#=
Some families might specify materials (particularly, those that are not backed up by a BIM tool)
We can retrieve those materials by querying the BIM shape.
=#

used_materials(s::BIMShape) = used_materials(s.family)

# CAD tools that support Layers might benefit from this typical implementation:
struct LayerFamily <: Family
  name::String
  color::RGB
  ref::Parameter{Any}
end

export LayerFamily, layer_family
layer_family(name, color::RGB=rgb(1,1,1)) =
  LayerFamily(name, color, Parameter{Any}(nothing))

backend_get_family_ref(b::Backend, f::Family, af::LayerFamily) =
  b_layer(b, af.name, true, af.color)



macro deffamily(name, parent, fields...)
  name_str = string(name)
  abstract_name = esc(Symbol(string))
  struct_name = esc(Symbol(string(map(uppercasefirst,split(name_str,'_'))...)))
  # We always add a name field
  fields = [:(name::String=$(name_str)), fields...]
  field_names = map(field -> field.args[1].args[1], fields)
  field_types = map(field -> field.args[1].args[2], fields)
  field_inits = map(field -> field.args[2], fields)
  field_renames = map(esc ∘ Symbol ∘ uppercasefirst ∘ string, field_names)
  field_replacements = Dict(zip(field_names, field_renames))
  struct_fields = map((name,typ) -> :($(name) :: $(typ)), field_names, field_types)
#  opt_params = map((name,typ,init) -> :($(name) :: $(typ) = $(init)), field_renames, field_types, field_inits)
#  key_params = map((name,typ,rename) -> :($(name) :: $(typ) = $(rename)), field_names, field_types, field_renames)
#  mk_param(name,typ) = Expr(:kw, Expr(:(::), name, typ))
  mk_param(name,typ,init) = Expr(:kw, Expr(:(::), name, typ), init)
  opt_params = map(mk_param, field_renames, field_types, map(init -> replace_in(init, field_replacements), field_inits))
  key_params = map(mk_param, field_names, field_types, field_renames)
  instance_params = map(mk_param, field_names, field_types, map(name -> :(family.$(name)), field_names))
  constructor_name = esc(name)
  instance_name = esc(Symbol(name_str, "_element")) #"_instance")) beam_family_element or beam_family_instance?
  default_name = esc(Symbol("default_", name_str))
  predicate_name = esc(Symbol("is_", name_str))
  selector_names = map(field_name -> esc(Symbol(name_str, "_", string(field_name))), field_names)
  with_name = esc(Symbol("with_", name_str))
  quote
    export $(constructor_name), $(instance_name), $(default_name), $(predicate_name), $(struct_name), $(with_name)
    struct $struct_name <: $parent
      $(struct_fields...)
      based_on::Union{Family, Nothing}
      implemented_as::IdDict{<:Backend, <:Family}
      ref::IdDict{<:Backend, Any}
      data::BackendParameter
    end
    $(constructor_name)($(opt_params...);
                        $(key_params...),
                        based_on=nothing,
                        implemented_as=IdDict{Backend, Family}(),
                        data=BackendParameter()) =
      $(struct_name)($(field_names...), based_on, implemented_as, IdDict{Backend, Any}(), data)
    $(instance_name)(family:: Family, implemented_as=copy(family.implemented_as); $(instance_params...)) =
      $(struct_name)($(field_names...), family, implemented_as, IdDict{Backend, Any}(), copy(family.data))
    $(default_name) = Parameter{$struct_name}($(constructor_name)())
    $(predicate_name)(v::$(struct_name)) = true
    $(predicate_name)(v::Any) = false
    $(with_name)(f::Function; family :: $(struct_name) = $(default_name)(), $(instance_params...)) =
      with(f, $(default_name), $(instance_name)(family; $([:($(esc(p))=$(esc(p))) for p in field_names]...)))
#    $(map((selector_name, field_name) -> :($(selector_name)(v::$(struct_name)) = v.$(field_name)),
#          selector_names, field_names)...)
    KhepriBase.meta_program(v::$(struct_name)) =
        Expr(:call, $(Expr(:quote, name)), $(map(field_name -> :(meta_program(v.$(field_name))), field_names)...))
    KhepriBase.meta_program(v::Parameter{$struct_name}) =
        Expr(:call, $(Expr(:quote, default_name)))
  end
end

export set_family
const set_family = set_on!

#family_ref(b::Backend, m::Family) = ref(b, m).value
#family_ref(b::Backend, s::BIMShape) = family_ref(b, s.family)

#=

We also need to register families so that we know, at runtime, which ones are available.

We need BIMElement
avaliable_families(element::BIMElement) => Array of families
default_family_parameter(element::BIMElement) => Parameter

ALSO

Give families names so that the interface can present them



We need to detect using traceability the Khepri primitive that is called. So we cannot exclude the
Khepri module.

=#

# When dispatching a BIM operation to a backend, we also need to dispatch the family

backend_family(b::Backend, family::Family) =
  get(family.implemented_as, b) do
    isnothing(family.based_on) ? # this is not a family_element (nor a derivation of a family_element)
      error("Family $(family) is missing the implementation for backend $(b)") :
      backend_family(b, family.based_on)
  end

copy_struct(s::T) where T = T([getfield(s, k) for k ∈ fieldnames(T)]...)

# Backends will install their own families on top of the default families, e.g.,
# set_backend_family(default_beam_family(), revit, revit_beam_family)
set_backend_family(family::Family, backend::Backend, backend_family::Family) =
  begin
    family.implemented_as[backend]=backend_family
    delete!(family.ref, backend) #force recreation
  end

realize(b::Backend, f::Family) =
  backend_get_family_ref(b, f, backend_family(b, f))

backend_get_family_ref(b::Backend, f::Family, bf) = bf
export backend_family, set_backend_family

#=
We can now define specific families for slabs, beams, etc., the corresponding
specific building elements and even, when possible, default implementations.
=#

@deffamily(slab_family, Family,
  thickness::Real=0.2,
  coating_thickness::Real=0.0,
  bottom_material::Material=material_concrete,
  top_material::Material=material_concrete,
  side_material::Material=material_concrete)

slab_family_elevation(b::Backend, family::SlabFamily) =
  family.coating_thickness - family.thickness
slab_family_thickness(b::Backend, family::SlabFamily) =
  family.coating_thickness + family.thickness

used_materials(f::SlabFamily) = (f.bottom_material, f.top_material, f.side_material)


@defproxy(slab, BIMShape, region::Region=rectangular_path(),
          level::Level=default_level(), family::SlabFamily=default_slab_family())

# Default implementation: dispatch on the slab elements
realize(b::Backend, s::Slab) =
  b_slab(b, s.region, s.level, s.family)

#=
export add_slab_opening
add_slab_opening(s::Slab=required(), contour::ClosedPath=circular_path()) =
    let b = backend(s)
        push!(s.openings, contour)
        if realized(s)
            set_ref!(s, realize_slab_openings(b, s, ref(b, s), [contour]))
        end
        s
    end

realize_slab_openings(b::Backend, s::Slab, s_ref, openings) =
    let s_base_height = s.level.height,
        s_thickness = slab_family_thickness(b, s.family)
        for opening in openings
            op_path = translate(opening, vz(s_base_height-1.1*s_thickness))
            op_ref = ensure_ref(b, backend_slab(b, op_path, s_thickness*1.2))
            s_ref = ensure_ref(b, subtract_ref(b, s_ref, op_ref))
        end
        s_ref
    end
=#

# Roof

@deffamily(roof_family, Family,
    thickness::Real=0.2,
    coating_thickness::Real=0.0,
    bottom_material::Material=material_concrete,
    top_material::Material=material_concrete,
    side_material::Material=material_concrete)

slab_family_elevation(b::Backend, family::RoofFamily) = 0
slab_family_thickness(b::Backend, family::RoofFamily) =
  family.coating_thickness + family.thickness

used_materials(f::RoofFamily) = (f.bottom_material, f.top_material, f.side_material)

@defproxy(roof, BIMShape, region::Region=rectangular_path(),
          level::Level=default_level(), family::RoofFamily=default_roof_family())
realize(b::Backend, s::Roof) =
  b_roof(b, s.region, s.level, s.family)

# Panel

@deffamily(panel_family, Family,
  thickness::Real=0.02,
  right_material::Material=material_glass,
  left_material::Material=material_glass,
  side_material::Material=material_glass)

used_materials(f::PanelFamily) = (f.right_material, f.left_material, f.side_material)


@defproxy(panel, BIMShape, region::Region=rectangular_path(), family::PanelFamily=default_panel_family())

realize(b::Backend, s::Panel) =
  b_panel(b, s.region, s.family)

#=

A wall contains doors and windows

=#

# Wall

@deffamily(wall_family, Family,
  thickness::Real=0.2,
  left_coating_thickness::Real=0.0,
  right_coating_thickness::Real=0.0,
  right_material::Material=material_plaster,
  left_material::Material=material_plaster,
  side_material::Material=material_plaster)

used_materials(f::WallFamily) = (f.right_material, f.left_material, f.side_material)

@defproxy(wall, BIMShape, path::Path=rectangular_path(),
          bottom_level::Level=default_level(),
          top_level::Level=upper_level(convert(Level, bottom_level)),
          family::WallFamily=default_wall_family(),
          offset::Real=is_closed_path(path) ? 1/2 : 0, # offset is relative to the thickness
          doors::Shapes=Shape[], windows::Shapes=Shape[])

# To deal with incremental creation, it is necessary to use transactions.
export with_wall
with_wall(f, args...) =
  with_transaction() do
    let w = wall(args...)
      f(w)
      w
    end
  end

# The protocol starts by identifying the approach to use. It can be either
# based on Boolean operations or on the construction of polygonal elements.
# Then, for each approach, the appropriate implementation is selected.
export HasBooleanOps, has_boolean_ops

struct HasBooleanOps{T} end
# By default, we rely on boolean operations
# By default, we DON'T rely on boolean operations
has_boolean_ops(::Type{<:Backend}) = HasBooleanOps{false}()

realize(b::B, w::Wall) where B<:Backend =
  realize(has_boolean_ops(B), b, w)

realize(::HasBooleanOps{true}, b::Backend, w::Wall) =
  with_material_as_layer(b, w.family) do
    realize_wall_openings(b, w, realize_wall_no_openings(b, w), [w.doors..., w.windows...])
  end

realize_wall_no_openings(b::Backend, w::Wall) =
  let w_base_height = w.bottom_level.height,
      w_height = w.top_level.height - w_base_height,
      w_path = translate(w.path, vz(w_base_height)),
      r_thickness = r_thickness(w),
      l_thickness = l_thickness(w)
    ensure_ref(b, backend_wall(b, w_path, w_height, l_thickness, r_thickness, w.family))
  end

realize_wall_openings(b::Backend, w::Wall, w_ref, openings) =
  let w_base_height = w.bottom_level.height,
      w_height = w.top_level.height - w_base_height,
      w_path = translate(w.path, vz(w_base_height)),
      r_thickness = r_thickness(w),
      l_thickness = l_thickness(w)
    for opening in openings
      w_ref = realize_wall_opening(b, w_ref, w_path, l_thickness, r_thickness, opening, w.family)
      ref(b, opening)
    end
    w_ref
  end

realize_wall_opening(b::Backend, w_ref, w_path, l_thickness, r_thickness, op, family) =
  let op_base_height = op.loc.y,
      op_height = op.family.height,
      op_path = translate(subpath(w_path, op.loc.x, op.loc.x + op.family.width), vz(op_base_height)),
      op_ref = ensure_ref(b, backend_wall(b, op_path, op_height, l_thickness+0.1, r_thickness+0.1, family))
    ensure_ref(b, subtract_ref(b, w_ref, op_ref))
  end

# For backends that do not support boolean operations, we use a different approach

realize(::HasBooleanOps{false}, b::Backend, w::Wall) =
  let w_base_height = w.bottom_level.height,
      w_height = w.top_level.height - w_base_height,
      r_thickness = r_thickness(w),
      l_thickness = l_thickness(w),
      w_path = translate(w.path, vz(w_base_height)),
      w_paths = subpaths(w_path),
      r_w_paths = subpaths(offset(w_path, -r_thickness)),
      l_w_paths = subpaths(offset(w_path, l_thickness)),
      openings = [w.doors..., w.windows...],
      prevlength = 0,
      matright = w.family.right_material,
      matleft = w.family.left_material,
      matside = w.family.side_material,
      refs = []
    for (w_seg_path, r_w_path, l_w_path) in zip(w_paths, r_w_paths, l_w_paths)
      let currlength = prevlength + path_length(w_seg_path),
          c_r_w_path = closed_path_for_height(r_w_path, w_height),
          c_l_w_path = closed_path_for_height(l_w_path, w_height)
        append!(refs, materialize_path(b, c_r_w_path, c_l_w_path, matside))
        openings = filter(openings) do op
          if prevlength <= op.loc.x < currlength ||
             prevlength <= op.loc.x + op.family.width <= currlength # contained (at least, partially)
            let op_height = op.family.height,
                op_at_start = op.loc.x <= prevlength,
                op_at_end = op.loc.x + op.family.width >= currlength,
                op_path = subpath(w_path,
                                  max(prevlength, op.loc.x),
                                  min(currlength, op.loc.x + op.family.width)),
                r_op_path = offset(op_path, -r_thickness),
                l_op_path = offset(op_path,  l_thickness),
                fixed_r_op_path =
                  open_polygonal_path([path_start(op_at_start ? r_w_path : r_op_path),
                                       path_end(op_at_end ? r_w_path : r_op_path)]),
                fixed_l_op_path =
                  open_polygonal_path([path_start(op_at_start ? l_w_path : l_op_path),
                                       path_end(op_at_end ? l_w_path : l_op_path)]),
                c_r_op_path = closed_path_for_height(translate(fixed_r_op_path, vz(op.loc.y)), op_height),
                c_l_op_path = closed_path_for_height(translate(fixed_l_op_path, vz(op.loc.y)), op_height)
              append!(refs, materialize_path(b, reverse(c_r_op_path), reverse(c_l_op_path), matside))
              c_r_w_path, c_l_w_path = subtract_paths(b, c_r_w_path, c_l_w_path, c_r_op_path, c_l_op_path)
              # preserve if not totally contained
              ! (op.loc.x >= prevlength && op.loc.x + op.family.width <= currlength)
            end
          else
            true
          end
        end
        prevlength = currlength
        push!(refs, materialize_path(b, reverse(c_l_w_path), matleft))
        push!(refs, materialize_path(b, c_r_w_path, matright))
      end
    end
    refs
  end

closed_offsetted_path(path, v) =
  let ps = path_vertices(path)
    closed_polygonal_path([ps..., reverse(map(p -> p+v, ps))...])
  end

closed_path_for_height(path, h) =
  closed_offsetted_path(path, vz(h))

subtract_paths(b::Backend, c_r_w_path, c_l_w_path, c_r_op_path, c_l_op_path) =
  region(c_r_w_path, c_r_op_path), region(c_l_w_path, c_l_op_path)
  #=
  let idxs = closest_vertices_indexes(path_vertices(c_r_w_path), path_vertices(c_r_op_path))
    closed_polygonal_path(
      inject_polygon_vertices_at_indexes(path_vertices(c_r_w_path), path_vertices(c_r_op_path), idxs)),
    closed_polygonal_path(
      inject_polygon_vertices_at_indexes(path_vertices(c_l_w_path), path_vertices(c_l_op_path), idxs))
  end=#

#=
Walls can be joined. That is very important because the wall needs to have
uniform thickness along the entire path.
=#
export join_walls
join_walls(wall1, wall2) =
  if wall1.bottom_level != wall2.bottom_level
    error("Walls with different bottom levels")
  elseif wall1.top_level != wall2.top_level
    error("Walls with different top levels")
  elseif wall1.family != wall2.family
    error("Walls with different families")
  elseif wall1.offset != wall2.offset
    error("Walls with different offsets")
  else
    let w = wall(join_paths(wall1.path, wall2.path),
                 wall1.bottom_level, wall1.top_level,
                 wall1.family, wall1.offset),
        len = path_length(wall1.path)
      for (es,l) in ((wall1.doors, 0), (wall2.doors, len))
        for e in es
          add_door(w, e.loc+vx(l), e.family)
        end
      end
      for (es,l) in ((wall1.windows, 0), (wall2.windows, len))
        for e in es
          add_window(w, e.loc+vx(l), e.family)
        end
      end
      for w in (wall1, wall2)
        delete_shapes(w.doors)
        delete_shapes(w.windows)
        delete_shape(w)
      end
      w
    end
  end

join_walls(walls...) =
  reduce(join_walls, walls)

# Right and Left considering observer looking along with curve direction
# a non-closed wall should have a wall offset of zero and a r_thickness and a l_thickness of 1/2*thickness
# a closed wall should have a wall offset of 1/2 and a r_thickness of zero and a l_thickness of 1*thickness
# Thus, we have:
r_thickness(offset, thickness) = (1/2 - offset)*thickness
l_thickness(offset, thickness) = (1/2 + offset)*thickness

r_thickness(w::Wall) = r_thickness(w.offset, w.family.thickness + w.family.right_coating_thickness)
l_thickness(w::Wall) = l_thickness(w.offset, w.family.thickness + w.family.left_coating_thickness)

# Door

@deffamily(door_family, Family,
  width::Real=1.0,
  height::Real=2.0,
  thickness::Real=0.05,
  right_material::Material=material_wood,
  left_material::Material=material_wood,
  side_material::Material=material_wood)

used_materials(f::DoorFamily) = (f.right_material, f.left_material, f.side_material)


@defproxy(door, BIMShape, wall::Wall=required(), loc::Loc=u0(), flip_x::Bool=false, flip_y::Bool=false, family::DoorFamily=default_door_family())

# Window

@deffamily(window_family, Family,
  width::Real=1.0,
  height::Real=1.0,
  thickness::Real=0.05,
  right_material::Material=material_glass,
  left_material::Material=material_glass,
  side_material::Material=material_glass)

used_materials(f::WindowFamily) = (f.right_material, f.left_material, f.side_material)


@defproxy(window, BIMShape, wall::Wall=required(), loc::Loc=u0(), flip_x::Bool=false, flip_y::Bool=false, family::WindowFamily=default_window_family())

realize(b::Backend, s::Union{Door, Window}) =
  let base_height = s.wall.bottom_level.height + s.loc.y,
      height = s.family.height,
      subpath = translate(subpath(s.wall.path, s.loc.x, s.loc.x + s.family.width), vz(base_height)),
      r_thickness = r_thickness(s.wall),
      l_thickness = l_thickness(s.wall),
      thickness = s.family.thickness
    b_wall(b, subpath, height, (l_thickness - r_thickness + thickness)/2, (r_thickness - l_thickness + thickness)/2,
           s.family)
  end
##

export add_door
add_door(w::Wall=required(), loc::Loc=u0(), family::DoorFamily=default_door_family()) =
  let d = door(w, loc, family=family)
    push!(w.doors, d)
    delete_shape(w)
    force_realize(w)
    w
  end

export add_window
add_window(w::Wall=required(), loc::Loc=u0(), family::WindowFamily=default_window_family()) =
  let d = window(w, loc, family=family)
    push!(w.windows, d)
    delete_shape(w)
    force_realize(w)
    w
  end

#=
A curtain wall is a special kind of wall that is made of a frame with windows.
=#

@deffamily(curtain_wall_frame_family, Family,
  width::Real=0.1,
  depth::Real=0.1,
  depth_offset::Real=0.25,
  right_material::Material=material_metal,
  left_material::Material=material_metal,
  side_material::Material=material_metal)

used_materials(f::CurtainWallFrameFamily) = (f.right_material, f.left_material, f.side_material)

@deffamily(curtain_wall_family, Family,
  max_panel_dx::Real=1,
  max_panel_dy::Real=2,
  panel::PanelFamily=panel_family(thickness=0.05),
  boundary_frame::CurtainWallFrameFamily=
    curtain_wall_frame_family(width=0.1,depth=0.1,depth_offset=0.25),
  mullion_frame::CurtainWallFrameFamily=
    curtain_wall_frame_family(width=0.08,depth=0.09,depth_offset=0.2),
  transom_frame::CurtainWallFrameFamily=
    curtain_wall_frame_family(width=0.06,depth=0.1,depth_offset=0.11))

used_materials(f::CurtainWallFamily) =
  (used_materials(f.boundary_frame)...,
   used_materials(f.mullion_frame)...,
   used_materials(f.transom_frame)...)

@defproxy(curtain_wall, BIMShape,
          path::Path=rectangular_path(),
          bottom_level::Level=default_level(),
          top_level::Level=upper_level(bottom_level),
          family::CurtainWallFamily=default_curtain_wall_family(),
          offset::Real=0.0)
curtain_wall(p0::Loc, p1::Loc;
     bottom_level::Level=default_level(),
     top_level::Level=upper_level(bottom_level),
     family::CurtainWallFamily=default_curtain_wall_family(),
     offset::Real=0.0) =
  curtain_wall([p0, p1], bottom_level=bottom_level, top_level=top_level,
         family=family, offset=offset)

realize(b::Backend, s::CurtainWall) =
  let th = s.family.panel.thickness,
      bfw = s.family.boundary_frame.width,
      bfd = s.family.boundary_frame.depth,
      bfdo = s.family.boundary_frame.depth_offset,
      mfw = s.family.mullion_frame.width,
      mfd = s.family.mullion_frame.depth,
      mdfo = s.family.mullion_frame.depth_offset,
      tfw = s.family.transom_frame.width,
      tfd = s.family.transom_frame.depth,
      tfdo = s.family.transom_frame.depth_offset,
      path = curtain_wall_path(b, s, s.family.panel),
      path_length = path_length(path),
      bottom = level_height(b, s.bottom_level),
      top = level_height(b, s.top_level),
      height = top - bottom,
      x_panels = ceil(Int, path_length/s.family.max_panel_dx),
      y_panels = ceil(Int, height/s.family.max_panel_dy),
      refs = []
    push!(refs, b_curtain_wall(b, s, subpath(path, bfw, path_length-bfw), bottom+bfw, height-2*bfw, th/2, th/2, :panel))
    push!(refs, b_curtain_wall(b, s, path, bottom, bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), :boundary_frame))
    push!(refs, b_curtain_wall(b, s, path, top-bfw, bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), :boundary_frame))
    push!(refs, b_curtain_wall(b, s, subpath(path, 0, bfw), bottom+bfw, height-2*bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), :boundary_frame))
    push!(refs, b_curtain_wall(b, s, subpath(path, path_length-bfw, path_length), bottom+bfw, height-2*bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), :boundary_frame))
    for i in 1:y_panels-1
      l = height/y_panels*i
      sub = subpath(path, bfw, path_length-bfw)
      push!(refs, b_curtain_wall(b, s, sub, bottom+l-tfw/2, tfw, l_thickness(tfdo, tfd), r_thickness(tfdo, tfd), :transom_frame))
    end
    for i in 1:x_panels-1
      l = path_length/x_panels*i
      push!(refs, b_curtain_wall(b, s, subpath(path, l-mfw/2, l+mfw/2), bottom+bfw, height-2*bfw, l_thickness(mdfo, mfd), r_thickness(mdfo, mfd), :mullion_frame))
    end
    [ensure_ref(b,r) for r in refs]
  end

# By default, curtain wall panels are planar
curtain_wall_path(b::Backend, s::CurtainWall, panel_family::Family) =
  s.path

curtain_wall_path(b::Backend, s::CurtainWall, panel_family::PanelFamily) =
  let path_length = path_length(s.path),
      x_panels = ceil(Int, path_length/s.family.max_panel_dx),
      pts = map(t->in_world(location_at_length(s.path, t)),
                division(0, path_length, x_panels))
    polygonal_path(pts)
  end

#
# We need to redefine the default method (maybe add an option to the macro to avoid defining the meta_program)
# This needs to be fixed for windows
#=
meta_program(w::Wall) =
    if isempty(w.doors)
        Expr(:call, :wall,
             meta_program(w.path),
             meta_program(w.bottom_level),
             meta_program(w.top_level),
             meta_program(w.family))
    else
        let door = w.doors[1]
            Expr(:call, :add_door,
                 meta_program(wall(w.path, w.bottom_level, w.top_level, w.family, w.doors[2:end], w.windows)),
                 meta_program(door.loc),
                 meta_program(door.family))
        end
    end
=#

# Beam
# Beams are mainly horizontal elements. By default, a beam is aligned along its top axis
@deffamily(beam_family, Family,
  profile::ClosedPath=top_aligned_rectangular_profile(1, 2),
  material::Material=material_metal)

used_materials(f::BeamFamily) = (f.material, )


@defproxy(beam, BIMShape, cb::Loc=u0(), h::Real=1, angle::Real=0, family::BeamFamily=default_beam_family())
beam(cb::Loc, ct::Loc, Angle::Real=0, Family::BeamFamily=default_beam_family(); angle::Real=Angle, family::BeamFamily=Family) =
    let (c, h) = position_and_height(cb, ct)
      beam(c, h, angle, family)
    end

realize(b::Backend, s::Beam) =
  b_beam(b, s.cb, s.h, s.angle, s.family)

# Column
# Columns are mainly vertical elements. A column has its center axis aligned with a line defined by two points

@deffamily(column_family, Family,
  profile::ClosedPath=rectangular_profile(0.2, 0.2),
  material::Material=material_concrete)

used_materials(f::ColumnFamily) = (f.material, )

@defproxy(free_column, BIMShape, cb::Loc=u0(), h::Real=1, angle::Real=0, family::ColumnFamily=default_column_family())
free_column(cb::Loc, ct::Loc, Angle::Real=0, Family::ColumnFamily=default_column_family(); angle::Real=Angle, family::ColumnFamily=Family) =
  let (c, h) = position_and_height(cb, ct)
    free_column(c, h, angle, family)
  end

realize(b::Backend, s::FreeColumn) =
  b_free_column(b, s.cb, s.h, s.angle, s.family)

@defproxy(column, BIMShape, cb::Loc=u0(), angle::Real=0,
  bottom_level::Level=default_level(), top_level::Level=upper_level(bottom_level),
  family::ColumnFamily=default_column_family())

realize(b::Backend, s::Column) =
  b_column(b, s.cb, s.angle, s.bottom_level, s.top_level, s.family)

# Tables and chairs

@deffamily(table_family, Family,
  length::Real=1.6,
  width::Real=0.9,
  height::Real=0.75,
  top_thickness::Real=0.05,
  leg_thickness::Real=0.05,
  material::Material=material_wood)

used_materials(f::TableFamily) = (f.material, )


@deffamily(chair_family, Family,
  length::Real=0.4,
  width::Real=0.4,
  height::Real=1.0,
  seat_height::Real=0.5,
  thickness::Real=0.05,
  material::Material=material_wood)

used_materials(f::ChairFamily) = (f.material, )

@deffamily(table_chair_family, Family,
  table_family::TableFamily=default_table_family(),
  chair_family::ChairFamily=default_chair_family(),
  chairs_top::Int=1,
  chairs_bottom::Int=1,
  chairs_right::Int=2,
  chairs_left::Int=2,
  spacing::Real=0.7)

used_materials(f::TableChairFamily) =
  (used_materials(f.table_family)..., used_materials(f.chair_family))


@defproxy(table, BIMShape, loc::Loc=u0(), level::Level=default_level(), family::TableFamily=default_table_family())
table(loc::Loc, angle::Real, level=default_level(), family::TableFamily=default_table_family()) =
  table(loc_from_o_phi(loc, angle), level, family)
realize(b::Backend, s::Table) =
  let tf = s.family
    b_table(b, add_z(s.loc, s.level.height),
            tf.length, tf.width, tf.height,
            tf.top_thickness, tf.leg_thickness,
            material_ref(b, tf.material))
  end

@defproxy(chair, BIMShape, loc::Loc=u0(), level::Level=default_level(), family::ChairFamily=default_chair_family())
chair(loc::Loc, angle::Real, level::Level=default_level(), family::ChairFamily=default_chair_family()) =
  chair(loc_from_o_phi(loc, angle), level, family)
realize(b::Backend, s::Chair) =
  let cf = s.family
    b_chair(b, add_z(s.loc, s.level.height),
            cf.length, cf.width, cf.height,
            cf.seat_height, cf.thickness,
            material_ref(b, cf.material))
  end

@defproxy(table_and_chairs, BIMShape, loc::Loc=u0(), level::Level=default_level(), family::TableChairFamily=default_table_chair_family())
table_and_chairs(loc::Loc, angle::Real, level::Level=default_level(), family::TableChairFamily=default_table_chair_family()) =
  table_and_chairs(loc_from_o_phi(loc, angle), level, family)
realize(b::Backend, s::TableAndChairs) =
  let f = s.family,
      tf = f.table_family,
      cf = f.chair_family,
      tmat = material_ref(b, tf.material),
      cmat = material_ref(b, cf.material)
    b_table_and_chairs(b,
      add_z(s.loc, s.level.height),
      p->b_table(b, p, tf.length, tf.width, tf.height, tf.top_thickness, tf.leg_thickness, tmat),
      p->b_chair(b, p, cf.length, cf.width, cf.height, cf.seat_height, cf.thickness, cmat),
      tf.length,
      tf.width,
      f.chairs_top,
      f.chairs_bottom,
      f.chairs_right,
      f.chairs_left,
      f.spacing)
  end

# Lights
# A pointlight has a fixed, inverse-square attenuation. Intensity is in Candela, range is irrelevant for physically correct lighting
@defproxy(pointlight, BIMShape, loc::Loc=z(3), color::RGB=rgb(1,1,1), intensity::Real=1500.0, range::Real=10, level::Level=default_level())

realize(b::Backend, s::Pointlight) =
  b_pointlight(b, add_z(s.loc, s.level.height), s.color, s.intensity, s.range)

@defproxy(spotlight, BIMShape, loc::Loc=z(3), dir::Vec=vz(-1), hotspot::Real=pi/4, falloff::Real=pi/3)

realize(b::Backend, s::Spotlight) =
  b_spotlight(b, s.loc, s.dir, s.hotspot, s.falloff)

@defproxy(ieslight, BIMShape, file::String=required(), loc::Loc=z(3), dir::Vec=vz(-1), alpha::Real=0, beta::Real=0, gamma::Real=0)

realize(b::Backend, s::Ieslight) =
  b_ieslight(b, s.file, s.loc, s.dir, s.alpha, s.beta, s.gamma)


#################################
# Node support
Base.@kwdef struct TrussNodeSupport
    ux::Bool=false
    uy::Bool=false
    uz::Bool=false
    rx::Bool=false
    ry::Bool=false
    rz::Bool=false
end

const truss_node_support = TrussNodeSupport

@deffamily(truss_node_family, Family,
    radius::Real=0.03,
    support::Any=false,
    material::Material=material_metal)

@deffamily(truss_bar_family, Family,
    radius::Real=0.02,
    inner_radius::Real=0,
    material::Material=material_metal)

truss_bar_family_cross_section_area(f::TrussBarFamily) =
  error("This should be computed by the backend family") #truss_bar_family_cross_section_area(back)

# We need a few families by default
export free_truss_node_family, fixed_truss_node_family, truss_node_support

free_truss_node_family =
  truss_node_family_element(default_truss_node_family(),
                            support=truss_node_support())
fixed_truss_node_family =
  truss_node_family_element(default_truss_node_family(),
                            support=truss_node_support(ux=true, uy=true, uz=true))


@defproxy(truss_node, BIMShape, p::Loc=u0(), family::TrussNodeFamily=default_truss_node_family())
@defproxy(truss_bar, BIMShape, p0::Loc=u0(), p1::Loc=u0(), angle::Real=0, family::TrussBarFamily=default_truss_bar_family())

realize(b::Backend, s::TrussNode) =
  truss_node_is_supported(s) ?
    [b_truss_node(b, s.p, s.family), b_truss_node_support(b, s.p, s.family)] :
    b_truss_node(b, s.p, s.family)

realize(b::Backend, s::TrussBar) =
  b_truss_bar(b, s.p0, s.p1, s.family)


export truss_nodes, truss_bars
truss_nodes(ps, family=default_truss_node_family()) =
  [truss_node(p, family) for p in ps]
truss_bars(ps, qs, family=default_truss_bar_family()) =
  [truss_bar(p, q, 0, family) for (p, q) in zip(ps, qs)]

export truss_node_is_supported
truss_node_is_supported(n) =
  let s = n.family.support
    s != false && (s.ux || s.uy || s.uz || s.rx || s.ry || s.rz)
  end

truss_bar_cross_section_area(s::TrussBar) =
  truss_bar_family_cross_section_area(family_ref(top_backend(), s.family))
truss_bar_volume(s::TrussBar) =
  truss_bar_cross_section_area(s)*distance(s.p0, s.p1)


# Should we merge coincident nodes?
export merge_coincident_truss_nodes,
       coincident_truss_nodes_distance,
       merge_coincident_truss_bars,
       maybe_merged_node,
       maybe_merged_bar

const merge_coincident_truss_nodes = Parameter(true)
const coincident_truss_nodes_distance = Parameter(1e-6)

maybe_merged_node(b::Backend, s::TrussNode) =
  # We are allowed to replace a node that we just created with one that already exists
  let epsilon = coincident_truss_nodes_distance(),
      p = s.p
    for n in b.truss_nodes
      if distance(n.p, p) < epsilon
        merge_coincident_truss_nodes() || error("Coincident nodes $(s) and $(n) at $(p)")
        return n
      end
    end
    b.realized(false)
    push!(b.truss_nodes, s)
    s
  end

# Should we merge coincident bars?
const merge_coincident_truss_bars = Parameter(true)

maybe_merged_bar(b::Backend, s::TrussBar) =
  # We are allowed to replace a bar that we just created with one that already exists
  let epsilon = coincident_truss_nodes_distance(),
      p0 = s.p0,
      p1 = s.p1
    for b in b.truss_bars
      if (distance(b.p0, p0) < epsilon && distance(b.p1, p1) < epsilon) ||
         (distance(b.p0, p1) < epsilon && distance(b.p1, p0) < epsilon)
        merge_coincident_truss_bars() || error("Coincident bars $(s) and $(n) at $(p0) and $(p1)")
        return b
      end
    end
    b.realized(false)
    push!(b.truss_bars, s)
    s
  end

# Many analysis tools prefer a simplified node-and-bar representation.
# To merge nodes and bars we need a different structure. Maybe we should merge
# this with the BIM information
export TrussNodeData, truss_node_data, TrussBarData, truss_bar_data

struct TrussNodeData
    id::Int
    loc::Loc
    family::Any
    load::Any
end

truss_node_data(id::Int, loc::Loc, family::Any, load::Vec) =
  TrussNodeData(id, loc, family, load)

#
struct TrussBarData
    id::Int
    node1::TrussNodeData
    node2::TrussNodeData
    rotation::Real
    family::Any
end

truss_bar_data(id::Int, node0::TrussNodeData, node1::TrussNodeData, rotation::Real, family::Any) =
  TrussBarData(id, node0, node1, rotation, family)

export process_nodes, process_bars
process_nodes(nodes, load=vz(0), loads_points=Dict()) =
  let point_loads = Dict()
    for k in keys(loads_points)
      for p in loads_points[k]
        point_loads[p] = k
      end
    end
    [truss_node_data(i, in_world(node.p), node.family, load + get(point_loads, node.p, vz(0)))
     for (i, node) in enumerate(nodes)]
  end
process_bars(bars, processed_nodes) =
  let epsilon = coincident_truss_nodes_distance(),
      node_data_near(loc) =
        begin
          for nd in processed_nodes
            distance(loc, nd.loc) < epsilon && return nd
          end
          error("Bar without corresponding node at location $(loc)!")
        end
    [truss_bar_data(
      i,
      node_data_near(in_world(bar.p0)),
      node_data_near(in_world(bar.p1)),
      bar.angle,
      bar.family)
     for (i, bar) in enumerate(bars)]
  end

# Analysis
@defcb truss_analysis(load::Vec=vz(-1e5), self_weight::Bool=false, point_loads::Dict=Dict())
@defcb truss_bars_volume()
@defcb node_displacement_function(res::Any)

export view_truss_deformation
view_truss_deformation(
  results::Any=nothing,
  visualizer::Backend=autocad;
  factor::Real=100) =
  let disp = node_displacement_function(results),
      b = top_backend()
    with(current_backend, visualizer) do
      for node in b.truss_node_data
        d = disp(node)*factor
        p = node.loc
        truss_node(p+d, family=node.family)
      end
      for bar in b.truss_bar_data
        let (node1, node2) = (bar.node1, bar.node2),
            (p1, p2) = (node1.loc, node2.loc),
            (d1, d2) = (disp(node1)*factor, disp(node2)*factor)
          truss_bar(p1+d1, p2+d2, family=bar.family)
        end
      end
    end
  end

export show_truss_deformation
show_truss_deformation(
    results::Any=nothing,
    visualizer::Backend=autocad;
    node_radius::Real=0.08, bar_radius::Real=0.02, factor::Real=100,
    deformation_name::String="Deformation",
    deformation_color::RGB=rgb(1, 0, 0),
    no_deformation_name::String="No deformation",
    no_deformation_color::RGB=rgb(0, 1, 0)) =
  let disp = node_displacement_function(results),
      b = top_backend()
    with(current_backend, visualizer) do
      delete_all_shapes()
      with(current_layer, create_layer(no_deformation_name, true, no_deformation_color)) do
        for node in b.truss_node_data
          p = node.loc
          sphere(p, node_radius)
        end
        for bar in b.truss_bar_data
          let (node1, node2) = (bar.node1, bar.node2),
              (p1, p2) = (node1.loc, node2.loc)
            cylinder(p1, bar_radius, p2)
          end
        end
      end
      with(current_layer, create_layer(deformation_name, true, deformation_color)) do
        for node in b.truss_node_data
          d = disp(node)*factor
          p = node.loc
          sphere(p+d, node_radius)
        end
        for bar in b.truss_bar_data
          let (node1, node2) = (bar.node1, bar.node2),
              (p1, p2) = (node1.loc, node2.loc),
              (d1, d2) = (disp(node1)*factor, disp(node2)*factor)
            cylinder(p1+d1, bar_radius, p2+d2)
          end
        end
      end
    end
  end

export max_displacement
max_displacement(results, b::Backend=top_backend()) =
  let disp = node_displacement_function(results)
    maximum(map(norm∘disp, b.truss_node_data))
  end

#=
bar_length_variation(bar, disp) =
  let (node1, node2) = (bar.node1, bar.node2),
      (p1, p2) = (node1.loc, node2.loc),
      pre_length = distance(p1, p2),
      (d1, d2) = disp.((node1, node2)),
      pos_length = distance(p1 + d1, p2 + d2)
    pos_length - pre_length
  end

show_stresses(results, b::Backend=autocad) =
  let disp = node_displacement_function(results),
      to255(x) = round(UInt8, x*255),
      (min, max) = extrema([bar_length_variation(bar, disp) for bar in frame3dd.truss_bar_data])
    for bar in frame3dd.truss_bar_data
      let diff = bar_length_variation(bar, disp),
          color = diff > 0 ?
            KhepriBase.rgb(diff/max, 0, 0) :
            KhepriBase.rgb(0, 0, diff/min),
          bar_radius = 0.05,
          (node1, node2) = (bar.node1, bar.node2),
          (p1, p2) = (node1.loc, node2.loc),
          (d1, d2) = disp.((node1, node2)),
          s = cylinder(p1+d1, bar_radius, p2+d2)
        KhepriBase.@remote(b, SetShapeColor(KhepriBase.ref(b, s).value, to255(KhepriBase.red(color)), to255(KhepriBase.green(color)), to255(KhepriBase.blue(color))))
      end
    end
  end
=#

@defcb lighting_analysis()

###################################
# BIM
@defcb all_levels()
@defcb all_walls()
@defcb all_walls_at_level(level)
@defcbs realize_beam_profile(s::BIMShape, profile::Path, cb::Loc, length::Real)
#@defcbs slab(profile, holes, thickness, family)
#@defcbs wall(path, height, l_thickness, r_thickness, family)
#@defcbs panel(bot::Locs, top::Locs, family)

####################################

#=
The typical family change is:

with(default_XPTO_family, XPTO_family_element(default_XPTO_family(), param=value)) do
  ...
end

We will simplify this pattern by allowing the following syntax:

with(XPTO_family, param=value) do
  ...
end
=#


family_profile(b::Backend, family) =
  family.profile
