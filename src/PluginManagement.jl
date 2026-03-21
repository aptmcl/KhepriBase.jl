export copy_plugin_files!, make_plugin_checker

copy_plugin_files!(dlls, src_folder, dst_folder) =
  for dll in dlls
    let src = joinpath(src_folder, dll),
        dst = joinpath(dst_folder, dll)
      rm(dst, force=true)
      cp(src, dst)
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
