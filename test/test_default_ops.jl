# test_default_ops.jl — rank-6 default b_* operation regression tests.
#
# Exercise the layered default operations against MockBackend (brought into scope
# by runtests.jl's `include("TestMockBackend.jl")` and `@import_backend_api`).
using Test

@testset "Default ops (rank 6)" begin
  @testset "b_unite_refs on empty/single/multi (BACKEND-6/CONSIST-5)" begin
    b = MockBackend()
    # An empty union (degenerate ContourPath/Region/Mesh reaching here via
    # b_stroke_unite) used to crash with "reducing over an empty collection";
    # now it yields the empty-ref identity new_refs(b).
    @test KhepriBase.b_unite_refs(b, MockId[]) == KhepriBase.new_refs(b)
    # Single element is returned unchanged (the bare scalar ref).
    @test KhepriBase.b_unite_refs(b, MockId[7]) == 7
    # Multiple elements are collected (default b_unite_ref = vcat).
    @test KhepriBase.b_unite_refs(b, MockId[7, 8, 9]) == MockId[7, 8, 9]
  end

  @testset "b_text / b_text_size degrade on non-ASCII & control chars (BACKEND-7)" begin
    b = MockBackend()
    m = KhepriBase.void_ref(b)
    @test haskey(KhepriBase.letter_glyph, '?')   # the .notdef fallback glyph exists
    # Unknown characters (accented, Greek, tab, newline) must degrade to '?'
    # instead of raising KeyError, in BOTH the draw and measure ops.
    for s in ("é", "café", "\t", "\n", "α")
      @test (KhepriBase.b_text(b, s, u0(), 1.0, m); true)
      @test (KhepriBase.b_text_size(b, s, 1.0, m); true)
    end
    # A single unknown char measures exactly like the '?' fallback.
    @test KhepriBase.b_text_size(b, "é", 1.0, m) == KhepriBase.b_text_size(b, "?", 1.0, m)
  end
end
