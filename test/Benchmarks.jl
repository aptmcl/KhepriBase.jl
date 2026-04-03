# Benchmarks.jl — Shared performance benchmark suite for Khepri backends
#
# Usage from a backend's test directory:
#   include(joinpath(pkgdir(KhepriBase), "test", "Benchmarks.jl"))
#   run_benchmarks()  # uses current backend
#
# Each benchmark reports: time (s), allocations (MB), RPC count, shapes/s

using KhepriBase

struct BenchmarkResult
  name::String
  time_s::Float64
  alloc_mb::Float64
  rpc_count::Int
  shapes::Int
end

throughput(r::BenchmarkResult) =
  r.time_s > 0 ? r.shapes / r.time_s : Inf

function run_benchmark(name::Symbol, f, n_shapes::Int)
  # Warmup: single shape to trigger compilation and opcode registration
  delete_all_shapes()
  reset_rpc_count!()
  try f(1) catch end
  delete_all_shapes()
  # Measure
  reset_rpc_count!()
  GC.gc()
  alloc_before = Base.gc_live_bytes()
  t = @elapsed f(n_shapes)
  alloc_after = Base.gc_live_bytes()
  rpcs = rpc_count[]
  alloc_mb = (alloc_after - alloc_before) / 1e6
  BenchmarkResult(string(name), t, alloc_mb, rpcs, n_shapes)
end

# Benchmark definitions

function bench_primitives_1k(n)
  for i in 1:n
    let r = i ÷ 32, c = i % 32
      box(xyz(c * 2.0, r * 2.0, 0), 1, 1, 1)
    end
  end
end

function bench_primitives_10k(n)
  for i in 1:n
    let r = i ÷ 100, c = i % 100
      sphere(xyz(c * 3.0, r * 3.0, 0), 1)
    end
  end
end

function bench_mixed_shapes(n)
  let per_type = n ÷ 4
    for i in 1:per_type
      box(xyz(i * 2.0, 0, 0), 1, 1, 1)
    end
    for i in 1:per_type
      sphere(xyz(i * 2.0, 4, 0), 1)
    end
    for i in 1:per_type
      cylinder(xyz(i * 2.0, 8, 0), 0.5, 2)
    end
    for i in 1:per_type
      cone(xyz(i * 2.0, 12, 0), 1, xyz(i * 2.0, 12, 2))
    end
  end
end

function bench_materials_shared(n)
  let mat = material(base_color=rgba(0.8, 0.2, 0.2, 1))
    for i in 1:n
      box(xyz((i % 32) * 2.0, (i ÷ 32) * 2.0, 0), 1, 1, 1, material=mat)
    end
  end
end

function bench_materials_unique(n)
  for i in 1:n
    let mat = material(base_color=rgba(i/n, 0.5, 1 - i/n, 1))
      box(xyz((i % 32) * 2.0, (i ÷ 32) * 2.0, 0), 1, 1, 1, material=mat)
    end
  end
end

function bench_wall_simple(n)
  for i in 1:n
    wall(open_polygonal_path([xyz(0, i * 2.0, 0), xyz(5, i * 2.0, 0)]))
  end
end

function bench_wall_openings(n)
  for i in 1:n
    let w = wall(open_polygonal_path([xyz(0, i * 3.0, 0), xyz(8, i * 3.0, 0)]))
      add_door(w, xy(1.5, 0))
      add_door(w, xy(4.0, 0))
      add_window(w, xy(6.5, 0.9))
    end
  end
end

function bench_floor_plan(n)
  # n controls number of spaces in the plan
  let plan = floor_plan()
    for i in 1:n
      let x0 = (i - 1) * 5.0
        space(plan, "Room$i",
              [xy(x0, 0), xy(x0 + 4, 0), xy(x0 + 4, 4), xy(x0, 4)])
      end
    end
    # Connect adjacent spaces with doors
    spaces = plan.spaces
    for i in 2:length(spaces)
      connect(plan, spaces[i-1], spaces[i], :door)
    end
    build(plan)
  end
end

function bench_delete_all(n)
  for i in 1:n
    box(xyz((i % 32) * 2.0, (i ÷ 32) * 2.0, 0), 1, 1, 1)
  end
  delete_all_shapes()
end

# Suite runner

const BENCHMARKS = [
  (:primitives_1k,     bench_primitives_1k,     1000),
  (:primitives_10k,    bench_primitives_10k,    10000),
  (:mixed_shapes,      bench_mixed_shapes,      2000),
  (:materials_shared,  bench_materials_shared,  1000),
  (:materials_unique,  bench_materials_unique,  1000),
  (:wall_simple,       bench_wall_simple,       50),
  (:wall_openings,     bench_wall_openings,     20),
  (:floor_plan,        bench_floor_plan,        10),
  (:delete_all,        bench_delete_all,        1000),
]

function run_benchmarks(; benchmarks=BENCHMARKS)
  results = BenchmarkResult[]
  println("\n", "="^78)
  println("  Khepri Performance Benchmarks")
  println("  Backend: ", join([backend_name(b) for b in current_backends()], ", "))
  println("="^78)
  println()
  println(rpad("Benchmark", 22), rpad("Time (s)", 12), rpad("RPCs", 8),
          rpad("Shapes", 8), rpad("Shapes/s", 12), "Alloc (MB)")
  println("-"^78)
  for (name, f, n) in benchmarks
    r = run_benchmark(name, f, n)
    push!(results, r)
    println(rpad(r.name, 22),
            rpad(round(r.time_s, digits=3), 12),
            rpad(r.rpc_count, 8),
            rpad(r.shapes, 8),
            rpad(round(throughput(r), digits=1), 12),
            round(r.alloc_mb, digits=1))
  end
  println("-"^78)
  let total_time = sum(r.time_s for r in results),
      total_rpcs = sum(r.rpc_count for r in results),
      total_shapes = sum(r.shapes for r in results)
    println(rpad("TOTAL", 22),
            rpad(round(total_time, digits=3), 12),
            rpad(total_rpcs, 8),
            rpad(total_shapes, 8),
            rpad(round(total_shapes / total_time, digits=1), 12))
  end
  println()
  delete_all_shapes()
  results
end

export run_benchmarks, run_benchmark, BenchmarkResult, throughput
