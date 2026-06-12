# ---- Infix Operators ----
# Julia precedence: ^ > / > |
# This matches: vertical stacking > depth composition > width composition

import Base: |, /, ^

"""
    a | b

Place two space descriptions side by side along the x-axis (`beside_x`).
Julia precedence: `|` binds loosest of the three layout operators.
"""
(|)(a::SpaceDesc, b::SpaceDesc) = beside_x(a, b)

"""
    a / b

Place two space descriptions front-to-back along the y-axis (`beside_y`).
Julia precedence: `/` binds tighter than `|` but looser than `^`.
"""
(/)(a::SpaceDesc, b::SpaceDesc) = beside_y(a, b)

"""
    a ^ b

Stack two space descriptions vertically (`above`): `b` is placed on top
of `a`, so the operands read bottom-to-top. Since `^` is right-associative
in Julia, `ground ^ first ^ second` is `above(ground, above(first, second))`
— still bottom-to-top, with `second` as the highest storey.
Julia precedence: `^` binds tightest of the three layout operators.
"""
(^)(a::SpaceDesc, b::SpaceDesc) = above(a, b)
