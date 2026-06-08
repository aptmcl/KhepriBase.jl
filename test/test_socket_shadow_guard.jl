# test_socket_shadow_guard.jl — recurrence guard for the Sockets.connect shadow.
#
# The Designs combinator `connect` was exported by KhepriBase and shadowed
# `Sockets.connect` inside the module, so a bare `connect(port)` in socket code
# silently resolved to the design combinator (see CONSTRAINTS-5; the symptom was
# band-aided with a `Sockets.`-qualified call). The root cause was fixed by
# renaming the combinator to `connect_spaces`. This guard keeps it from coming
# back and catches any new bare, unqualified `connect(` in the kernel source.
using Test
using KhepriBase

@testset "Sockets-shadow guard (connect)" begin
  # `connect` must NOT be an exported KhepriBase name (it would re-shadow
  # Sockets.connect); the renamed combinator stays public.
  @test !(:connect in names(KhepriBase))
  @test :connect_spaces in names(KhepriBase)

  # No bare, unqualified `connect(` in src — only `Sockets.connect(` (qualified,
  # excluded by the lookbehind on `.`) and `connect_*(` combinators are allowed.
  srcdir = joinpath(dirname(pathof(KhepriBase)), "..", "src")
  bare_connect = r"(?<![.\w])connect\("
  offenders = String[]
  for f in filter(p -> endswith(p, ".jl"), readdir(srcdir, join=true))
    for (i, line) in enumerate(eachline(f))
      occursin(bare_connect, line) && push!(offenders, "$(basename(f)):$i")
    end
  end
  @test isempty(offenders)
end
