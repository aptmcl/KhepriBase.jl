public copy_plugin_files!, make_plugin_checker

copy_plugin_files!(dlls, src_folder, dst_folder) =
  for dll in dlls
    let src = joinpath(src_folder, dll),
        dst = joinpath(dst_folder, dll)
      # Verify the source before touching the destination — otherwise a missing/half-built src leaves the
      # installed plugin deleted (cp force=true still rm's dst before it stats src, so the guard is needed).
      isfile(src) || error("copy_plugin_files!: source not found, refusing to overwrite $dst: $src")
      cp(src, dst, force=true)
    end
  end

function make_plugin_checker(app_name, update_fn; retries=10, delay=5)
  checked = Ref(false)
  () -> begin
    checked[] && return
    @info("Checking $app_name plugin...")
    for i in 1:retries
      try
        update_fn()
        @info("done.")
        checked[] = true
        return
      catch exc
        if isa(exc, Base.IOError) && i < retries
          @error("The $app_name plugin is outdated! Please, close $app_name.")
          sleep(delay)
        else
          rethrow()
        end
      end
    end
  end
end
