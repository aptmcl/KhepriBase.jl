# ============================================================
# Agent-based simulation operations
#
# These operations have no generic fallback — they require a
# supporting backend (e.g. KhepriUnity).  Non-supporting
# backends throw UndefinedBackendException automatically.
# ============================================================

# NavMesh
@defcb enable_nav_mesh_tagging(enable::Bool)
@defcb set_nav_mesh_area(shape, area)
@defcb update_nav_mesh()

# Convenience: bracket geometry creation with NavMesh tagging + bake.
# Walls are tagged not-walkable, slabs walkable (backend-specific).
export with_simulation
with_simulation(f) =
  begin
    enable_nav_mesh_tagging(true)
    try
      f()
      update_nav_mesh()
    finally
      enable_nav_mesh_tagging(false)
    end
  end

# Movement model
@defcb set_sim_hsf(relaxation_time=0.5, max_speed_coef=1.3,
                   V=2.1, sigma=0.3, U=10.0, R=0.2, c=0.5, phi=100.0)
@defcb set_sim_none()

# Velocity distribution
@defcb set_vel_gauss_hsf(mean, std_dev, min_v, max_v)
@defcb set_vel_uniform_hsf(min_v, max_v)
@defcb set_vel_hsf(vel)

# Goals and agents
@defcb create_sim_goal(pos, scale, rot)
@defcb create_sim_agent(pos, rot, goal_ids, color=0xb73478)
@defcb spawn_agents_rect(count, center, dx, dz, rot, goal_ids, color=0xb73478)
@defcb spawn_agents_ellipse(count, center, dx, dz, rot, goal_ids, color=0xb73478)
@defcb spawn_agents_polygon(count, h, vertices, goal_ids, color=0xb73478)

# Simulation control
@defcb start_simulation(max_time)
@defcb set_simulation_speed(speed)
@defcb is_simulation_finished()
@defcb was_simulation_successful()
@defcb get_evacuation_time()
@defcb reset_simulation()
@defcb set_sim_rand_seed(seed)

# Blocking simulation: starts, waits for completion, returns evacuation time.
@defcb run_simulation(max_time)
