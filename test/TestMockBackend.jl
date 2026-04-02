# TestMockBackend.jl - Minimal mock backend for testing Khepri without external applications

# Guard to prevent multiple inclusions
if !@isdefined(MockBackend)

using KhepriBase
import KhepriBase: Backend, realize, void_ref, new_refs, b_trig, b_point, b_line,
  b_polygon, b_circle, b_arc, b_spline, b_closed_spline, b_rectangle,
  b_surface_polygon, b_surface_circle, b_surface_rectangle,
  b_box, b_sphere, b_cylinder, b_cone, b_torus, b_delete_ref, b_delete_refs,
  b_all_shape_refs, b_created_shape_refs, b_created_shapes,
  b_existing_shape_refs, b_existing_shapes,
  shape_storage_type, ShapeStorageType, LocalShapeStorage, RemoteShapeStorage,
  current_transaction, Transaction, AutoCommitTransaction,
  GenericRef, NativeRef, NativeRefs, References, backend_name

export MockBackend, mock_backend, reset_mock_backend!, mock_geometry_stats

# Type keys for the mock backend
struct MockKey end

# Reference type for mock backend
const MockId = Int

# Geometry record types for tracking created geometry
struct MockPoint
  position::Loc
end

struct MockLine
  vertices::Vector{<:Loc}
end

struct MockTriangle
  p1::Loc
  p2::Loc
  p3::Loc
end

struct MockCircle
  center::Loc
  radius::Real
end

struct MockArc
  center::Loc
  radius::Real
  start_angle::Real
  amplitude::Real
end

struct MockSphere
  center::Loc
  radius::Real
end

struct MockBox
  corner::Loc
  dx::Real
  dy::Real
  dz::Real
end

struct MockCylinder
  center::Loc
  radius::Real
  height::Real
end

# Mock backend structure
mutable struct MockBackend <: Backend{MockKey, MockId}
  # Geometry storage
  points::Vector{MockPoint}
  lines::Vector{MockLine}
  triangles::Vector{MockTriangle}
  circles::Vector{MockCircle}
  arcs::Vector{MockArc}
  spheres::Vector{MockSphere}
  boxes::Vector{MockBox}
  cylinders::Vector{MockCylinder}

  # All refs for tracking
  all_refs::Vector{MockId}

  # Reference counter
  next_id::Int

  # Required backend fields
  transaction::Parameter{Transaction}
  refs::References{MockKey, MockId}
end

# Constructor
function MockBackend()
  MockBackend(
    MockPoint[],
    MockLine[],
    MockTriangle[],
    MockCircle[],
    MockArc[],
    MockSphere[],
    MockBox[],
    MockCylinder[],
    MockId[],
    1,
    Parameter{Transaction}(AutoCommitTransaction()),
    References{MockKey, MockId}()
  )
end

# Backend protocol implementations
KhepriBase.current_transaction(b::MockBackend) = b.transaction
KhepriBase.backend_name(b::MockBackend) = "MockBackend"

# Generate next reference ID
function next_ref!(b::MockBackend)
  id = b.next_id
  b.next_id += 1
  push!(b.all_refs, id)
  id
end

# void_ref returns a special "nothing" reference
void_ref(b::MockBackend) = 0

# new_refs returns an empty vector of the backend's reference type
new_refs(b::MockBackend) = MockId[]

# Get all shape refs
b_all_shape_refs(b::MockBackend) = copy(b.all_refs)

# Delete operations
b_delete_ref(b::MockBackend, r::MockId) = filter!(x -> x != r, b.all_refs)
b_delete_refs(b::MockBackend, rs::Vector{MockId}) = filter!(x -> !(x in rs), b.all_refs)

# Reset mock backend state
function reset_mock_backend!(b::MockBackend)
  empty!(b.points)
  empty!(b.lines)
  empty!(b.triangles)
  empty!(b.circles)
  empty!(b.arcs)
  empty!(b.spheres)
  empty!(b.boxes)
  empty!(b.cylinders)
  empty!(b.all_refs)
  empty!(b.refs.shapes)
  empty!(b.refs.materials)
  empty!(b.refs.layers)
  empty!(b.refs.annotations)
  empty!(b.refs.families)
  empty!(b.refs.levels)
  b.next_id = 1
  b
end

# Get geometry statistics
function mock_geometry_stats(b::MockBackend)
  (
    points = length(b.points),
    lines = length(b.lines),
    triangles = length(b.triangles),
    circles = length(b.circles),
    arcs = length(b.arcs),
    spheres = length(b.spheres),
    boxes = length(b.boxes),
    cylinders = length(b.cylinders),
    total_refs = length(b.all_refs)
  )
end

# Tier 0: Points and curves
b_point(b::MockBackend, p, mat) = begin
  push!(b.points, MockPoint(p))
  next_ref!(b)
end

b_line(b::MockBackend, ps, mat) = begin
  push!(b.lines, MockLine(collect(ps)))
  next_ref!(b)
end

b_polygon(b::MockBackend, ps, mat) = begin
  push!(b.lines, MockLine([ps..., ps[1]]))
  next_ref!(b)
end

b_rectangle(b::MockBackend, c, dx, dy, mat) = begin
  push!(b.lines, MockLine([c, c+vx(dx), c+vxy(dx, dy), c+vy(dy), c]))
  next_ref!(b)
end

b_circle(b::MockBackend, c, r, mat) = begin
  push!(b.circles, MockCircle(c, r))
  next_ref!(b)
end

b_arc(b::MockBackend, c, r, α, Δα, mat) = begin
  push!(b.arcs, MockArc(c, r, α, Δα))
  next_ref!(b)
end

b_spline(b::MockBackend, ps, v1, v2, mat) = begin
  push!(b.lines, MockLine(collect(ps)))
  next_ref!(b)
end

b_closed_spline(b::MockBackend, ps, mat) = begin
  push!(b.lines, MockLine([ps..., ps[1]]))
  next_ref!(b)
end

# Tier 1: Triangles (fundamental)
b_trig(b::MockBackend, p1, p2, p3) = begin
  push!(b.triangles, MockTriangle(p1, p2, p3))
  next_ref!(b)
end

b_trig(b::MockBackend, p1, p2, p3, mat) = b_trig(b, p1, p2, p3)

# Tier 2: Surfaces (use default implementations based on b_trig)
# b_surface_polygon, b_surface_circle, etc. will use the default fallbacks

# Tier 3: Solids
b_sphere(b::MockBackend, c, r, mat) = begin
  push!(b.spheres, MockSphere(c, r))
  next_ref!(b)
end

b_box(b::MockBackend, c, dx, dy, dz, mat) = begin
  push!(b.boxes, MockBox(c, dx, dy, dz))
  next_ref!(b)
end

b_cylinder(b::MockBackend, cb, r, h, bmat, tmat, smat) = begin
  push!(b.cylinders, MockCylinder(cb, r, h))
  next_ref!(b)
end

# Layer operations (minimal implementation)
KhepriBase.b_layer(b::MockBackend, name, visible, color) = next_ref!(b)
KhepriBase.b_current_layer_ref(b::MockBackend) = 0
KhepriBase.b_current_layer_ref(b::MockBackend, r) = nothing
KhepriBase.b_delete_all_shapes_in_layer(b::MockBackend, layer) = nothing
KhepriBase.b_create_layer_from_ref_value(b::MockBackend, r) = layer("Default")

# Material operations (minimal implementation)
KhepriBase.b_get_material(b::MockBackend, spec::Nothing) = 0
KhepriBase.b_get_material(b::MockBackend, spec) = 0
KhepriBase.b_get_material(b::MockBackend, ::BackendDefault) = 0
KhepriBase.b_material(b::MockBackend, name, base_color) =
  next_ref!(b)

# Create global mock backend instance
const _mock_backend = Ref{Union{Nothing, MockBackend}}(nothing)

function mock_backend()
  if isnothing(_mock_backend[])
    _mock_backend[] = MockBackend()
  end
  _mock_backend[]
end

# Function to use mock backend as current backend
function with_mock_backend(f)
  b = mock_backend()
  reset_mock_backend!(b)
  with(current_backend, b) do
    f(b)
  end
end

export with_mock_backend

end # if !@isdefined(MockBackend)
