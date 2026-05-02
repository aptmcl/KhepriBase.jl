# stress/exercises.jl — solutions from teaching exercises
#
# Tests built from the answer blocks in
#   ~/Vault/AML/Teaching/ElementosPedagogicos/Arquitectura/aulas-praticas-julia/
# (aula-NN.tex and map15-* / map30-* exercise sheets).
#
# Each test reproduces one exercise solution verbatim — wrapped in a
# closure that defines the helpers locally and ends with a top-level call.
# Where the printed solution lacked a top-level call, a synthetic one is
# added with sensible parameters.
#
# Tests are ordered by source file (aula-02 → aula-11, then map15-*,
# then map30-*). Order matches the lecture sequence, so failures earlier
# in the suite indicate gaps in foundational coverage.
#
# A failure here may mean a backend bug OR a bug in the solution itself
# (the .tex files are sometimes copy-paste artefacts, REPL-style with
# out-of-order definitions, or contain syntax that no longer parses).
# Treat each failure as suspect until traced.

stress_exercises(b, reset!, verify) =
  @testset "Exercises" begin
    reset!()
    slot = Slot(:exercises, 0, 0)

    # ── aula-02: circles and basic 2D layouts ────────────────────────

    run_one_test(b, slot, "aula02_two_circles",
      () -> begin
        circle(xy(-1, 0), 1)
        circle(xy(+1, 0), 1)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula02_four_circles",
      () -> begin
        circle(xy(-1, -1), 1)
        circle(xy(-1, +1), 1)
        circle(xy(+1, +1), 1)
        circle(xy(+1, -1), 1)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula02_three_circles_polar",
      () -> begin
        circle(pol(1.1547, 0/3*pi), 1)
        circle(pol(1.1547, 2/3*pi), 1)
        circle(pol(1.1547, 4/3*pi), 1)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula02_circulo_e_raio",
      () -> begin
        circulo_e_raio(p, r) =
          begin
            circle(p, r)
            text(string("Raio: ", r), p + vpol(r*1.25, 0), r*0.25)
          end
        circulo_e_raio(pol(0, 0), 4)
        circulo_e_raio(pol(4, pi/4), 2)
        circulo_e_raio(pol(6, pi/4), 1)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula02_seta",
      () -> begin
        seta(p, ro, alfa, sigma, beta) =
          line(p,
               p + vpol(ro, alfa),
               p + vpol(ro, alfa) + vpol(sigma, alfa + pi - beta),
               p + vpol(ro, alfa) + vpol(sigma, alfa + pi + beta),
               p + vpol(ro, alfa))
        seta(u0(), 5.0, π/3, 1.0, π/8)
      end,
      nothing,
      verify)

    # ── aula-03: recursive 2D compositions ───────────────────────────

    run_one_test(b, slot, "aula03_equilibrio_circulos",
      () -> begin
        equilibrio_circulos(p, r, f) =
          if r < 1
            nothing
          else
            circle(p, r)
            equilibrio_circulos(p + vy((1 + f)*r), f*r, f)
          end
        equilibrio_circulos(u0(), 5.0, 0.7)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula03_flor",
      () -> begin
        circulos_radiais(p, n, r0, r1, fi, d_fi) =
          if n == 0
            nothing
          else
            circle(p + vpol(r0, fi), r1)
            circulos_radiais(p, n - 1, r0, r1, fi + d_fi, d_fi)
          end
        flor(p, raio, petalas) =
          let coef = sin(pi/petalas)
            r1 = (raio*coef)/(1 - coef)
            r0 = raio + r1
            circle(p, raio)
            circulos_radiais(p, petalas, r0, r1, 0, 2*pi/petalas)
          end
        flor(xy(0, 0), 5, 10)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula03_serra",
      () -> begin
        serra(p0, dentes, comprimento, altura) =
          if dentes == 0
            nothing
          else
            let p1 = p0 + vxy(comprimento/2, altura)
                p2 = p0 + vx(comprimento)
                line(p0, p1, p2)
                serra(p2, dentes - 1, comprimento, altura)
            end
          end
        serra(u0(), 8, 1.0, 0.5)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula03_circulos_radiais",
      () -> begin
        circulos_radiais(p, n, r0, r1, fi, d_fi) =
          if n == 0
            nothing
          else
            circle(p + vpol(r0, fi), r1)
            circulos_radiais(p, n - 1, r0, r1, fi + d_fi, d_fi)
          end
        circulos_radiais(u0(), 12, 5.0, 1.0, 0.0, π/6)
      end,
      nothing,
      verify)

    # ── aula-04: simple 3D building grid ─────────────────────────────

    run_one_test(b, slot, "aula04_malha_predios",
      () -> begin
        predio(p, l, h) = box(p, l, l, h)
        rua_predios(p, m_predios, l, h, s) =
          if m_predios == 0
            nothing
          else
            predio(p, l, h)
            rua_predios(p + vx(l + s), m_predios - 1, l, h, s)
          end
        malha_predios(p, n_ruas, m_predios, l, h, s) =
          if n_ruas == 0
            nothing
          else
            rua_predios(p, m_predios, l, h, s)
            malha_predios(p + vy(l + s), n_ruas - 1, m_predios, l, h, s)
          end
        malha_predios(u0(), 4, 4, 5.0, 10.0, 2.0)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula04_predio_misto",
      () -> begin
        predio0(p, l, h) =
          box(p, l, l, random_range(0.1, 1.0)*h)
        predio1(p, l, h) =
          cylinder(p + vxy(l/2, l/2), l/2, random_range(0.1, 1.0)*h)
        predio(p, l, h) =
          random(5) == 0 ? predio1(p, l, h) : predio0(p, l, h)
        rua_predios(p, m, l, h, s) =
          if m == 0; nothing
          else; predio(p, l, h); rua_predios(p + vx(l + s), m - 1, l, h, s); end
        malha_predios(p, n, m, l, h, s) =
          if n == 0; nothing
          else; rua_predios(p, m, l, h, s); malha_predios(p + vy(l + s), n - 1, m, l, h, s); end
        malha_predios(u0(), 3, 3, 5.0, 10.0, 2.0)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula04_predio_random_altura",
      () -> begin
        predio(p, l, h) =
          box(p, l, l, random_range(0.1, 1.0)*h)
        rua_predios(p, m, l, h, s) =
          m == 0 ? nothing :
          begin predio(p, l, h); rua_predios(p + vx(l + s), m - 1, l, h, s) end
        malha_predios(p, n, m, l, h, s) =
          n == 0 ? nothing :
          begin rua_predios(p, m, l, h, s); malha_predios(p + vy(l + s), n - 1, m, l, h, s) end
        set_random_seed(12345)
        malha_predios(u0(), 4, 4, 5.0, 10.0, 2.0)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula04_predio_blocos",
      () -> begin
        predio_blocos(n, p0, c0, l0, h0) =
          if n == 1
            box(p0, c0, l0, h0)
          else
            let c1 = random_range(0.7, 1.0)*c0,
                l1 = random_range(0.7, 1.0)*l0,
                h1 = random_range(0.2, 0.8)*h0,
                p1 = p0 + vxyz((c0 - c1)/2, (l0 - l1)/2, h1)
              box(p0, c0, l0, h1)
              predio_blocos(n - 1, p1, c1, l1, h0 - h1)
            end
          end
        predio_blocos(4, u0(), 5.0, 5.0, 12.0)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula04_predio_gaussiana",
      () -> begin
        gaussiana_2d(x, y, sigma) = exp(-((x/sigma)^2 + (y/sigma)^2))
        predio0(p, l, h) = box(p, l, l, random_range(0.1, 1.0)*h)
        predio1(p, l, h) = cylinder(p + vxy(l/2, l/2), l/2, random_range(0.1, 1.0)*h)
        predio(p, l, h) =
          let h = h*max(0.1, gaussiana_2d(cx(p), cy(p), 25.0*l))
            random(5) == 0 ? predio1(p, l, h) : predio0(p, l, h)
          end
        rua_predios(p, m, l, h, s) =
          m == 0 ? nothing :
          begin predio(p, l, h); rua_predios(p + vx(l + s), m - 1, l, h, s) end
        malha_predios(p, n, m, l, h, s) =
          n == 0 ? nothing :
          begin rua_predios(p, m, l, h, s); malha_predios(p + vy(l + s), n - 1, m, l, h, s) end
        set_random_seed(12345)
        malha_predios(xyz(-50, -50, 0), 4, 4, 5.0, 10.0, 2.0)
      end,
      nothing,
      verify)

    # ── aula-05: trusses (spheres + cylinders) ───────────────────────

    run_one_test(b, slot, "aula05_trelica_recta",
      () -> begin
        raio_no_trelica = 0.1
        no_trelica(p) = sphere(p, raio_no_trelica)
        raio_barra_trelica = 0.03
        barra_trelica(p0, p1) = cylinder(p0, raio_barra_trelica, p1)
        nos_trelica(ps) = for p in ps; no_trelica(p) end
        barras_trelica(ps, qs) = for (p, q) in zip(ps, qs); barra_trelica(p, q) end
        trelica(ais, bis, cis) = begin
          nos_trelica(ais); nos_trelica(bis); nos_trelica(cis)
          barras_trelica(ais, cis)
          barras_trelica(bis, ais)
          barras_trelica(bis, cis)
          barras_trelica(bis, ais[2:end])
          barras_trelica(bis, cis[2:end])
          barras_trelica(ais[2:end], ais)
          barras_trelica(cis[2:end], cis)
          barras_trelica(bis[2:end], bis)
        end
        coordenadas_x(p, l, n) =
          n == 0 ? Loc[] : [p, coordenadas_x(p + vx(l), l, n - 1)...]
        trelica_recta(p, h, l, n) =
          trelica(coordenadas_x(p, l, n),
                  coordenadas_x(p + vxyz(l/2, l/2, h), l, n - 1),
                  coordenadas_x(p + vy(l), l, n))
        trelica_recta(u0(), 1.0, 1.0, 8)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula05_genoma",
      () -> begin
        raio_no_trelica = 0.1
        no_trelica(p) = sphere(p, raio_no_trelica)
        raio_barra_trelica = 0.03
        barra_trelica(p0, p1) = cylinder(p0, raio_barra_trelica, p1)
        nos_trelica(ps) = for p in ps; no_trelica(p) end
        barras_trelica(ps, qs) = for (p, q) in zip(ps, qs); barra_trelica(p, q) end
        escada_de_mao(ais, bis) = begin
          nos_trelica(ais); nos_trelica(bis)
          barras_trelica(ais, bis)
          barras_trelica(ais, ais[2:end])
          barras_trelica(bis, bis[2:end])
        end
        coordenadas_helice(p, r, fi, dfi, dz, n) =
          n == 0 ? Loc[] :
            [p + vpol(r, fi),
             coordenadas_helice(p + vz(dz), r, fi + dfi, dfi, dz, n - 1)...]
        genoma(p, r, dfi, dz, n) =
          escada_de_mao(coordenadas_helice(p, r, 0, dfi, dz, n),
                        coordenadas_helice(p, r, pi, dfi, dz, n))
        genoma(u0(), 1.0, π/16, 0.5, 12)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula05_escada_de_mao",
      () -> begin
        no_trelica(p) = sphere(p, 0.1)
        barra_trelica(p0, p1) = cylinder(p0, 0.03, p1)
        nos_trelica(ps) = for p in ps; no_trelica(p) end
        barras_trelica(ps, qs) = for (p, q) in zip(ps, qs); barra_trelica(p, q) end
        escada_de_mao(ais, bis) = begin
          nos_trelica(ais); nos_trelica(bis)
          barras_trelica(ais, bis)
          barras_trelica(ais, ais[2:end])
          barras_trelica(bis, bis[2:end])
        end
        escada_de_mao([xyz(0, 0, i*0.4) for i in 0:7],
                      [xyz(1, 0, i*0.4) for i in 0:7])
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula05_trelica_plana",
      () -> begin
        no_trelica(p) = sphere(p, 0.1)
        barra_trelica(p0, p1) = cylinder(p0, 0.03, p1)
        nos_trelica(ps) = for p in ps; no_trelica(p) end
        barras_trelica(ps, qs) = for (p, q) in zip(ps, qs); barra_trelica(p, q) end
        trelica_plana(ais, bis) = begin
          nos_trelica(ais); nos_trelica(bis)
          barras_trelica(ais, bis)
          barras_trelica(bis, ais[2:end])
          barras_trelica(ais, ais[2:end])
          barras_trelica(bis, bis[2:end])
        end
        coordenadas_x(p, l, n) =
          n == 0 ? Loc[] : [p, coordenadas_x(p + vx(l), l, n - 1)...]
        trelica_plana(coordenadas_x(u0(), 2.0, 8),
                      coordenadas_x(xyz(1, 0, 1), 2.0, 7))
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula05_trelica_especial",
      () -> begin
        no_trelica(p) = sphere(p, 0.1)
        barra_trelica(p0, p1) = cylinder(p0, 0.03, p1)
        nos_trelica(ps) = for p in ps; no_trelica(p) end
        barras_trelica(ps, qs) = for (p, q) in zip(ps, qs); barra_trelica(p, q) end
        ponto_medio(p, q) = xyz((cx(p) + cx(q))/2, (cy(p) + cy(q))/2, (cz(p) + cz(q))/2)
        pontos_medios(ps, qs) = [ponto_medio(p, q) for (p, q) in zip(ps, qs)]
        trelica_especial(ais, bis, cis) =
          let dis = pontos_medios(ais, cis)
            nos_trelica(ais); nos_trelica(bis); nos_trelica(cis); nos_trelica(dis)
            barras_trelica(ais, cis)
            barras_trelica(bis, dis)
            barras_trelica(bis, dis[2:end])
            barras_trelica(ais, ais[2:end])
            barras_trelica(cis, cis[2:end])
            barras_trelica(bis, bis[2:end])
          end
        trelica_especial(
          [xyz(i, 0, 0) for i in 0:5],
          [xyz(i + 0.5, 0.5, -0.7) for i in 0:4],
          [xyz(i, 1, 0) for i in 0:5])
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula05_trelica_especial_recta",
      () -> begin
        no_trelica(p) = sphere(p, 0.1)
        barra_trelica(p0, p1) = cylinder(p0, 0.03, p1)
        nos_trelica(ps) = for p in ps; no_trelica(p) end
        barras_trelica(ps, qs) = for (p, q) in zip(ps, qs); barra_trelica(p, q) end
        ponto_medio(p, q) = xyz((cx(p) + cx(q))/2, (cy(p) + cy(q))/2, (cz(p) + cz(q))/2)
        pontos_medios(ps, qs) = [ponto_medio(p, q) for (p, q) in zip(ps, qs)]
        trelica_especial(ais, bis, cis) =
          let dis = pontos_medios(ais, cis)
            nos_trelica(ais); nos_trelica(bis); nos_trelica(cis); nos_trelica(dis)
            barras_trelica(ais, cis)
            barras_trelica(bis, dis)
            barras_trelica(bis, dis[2:end])
            barras_trelica(ais, ais[2:end])
            barras_trelica(cis, cis[2:end])
            barras_trelica(bis, bis[2:end])
          end
        coordenadas_x(p, l, n) =
          n == 0 ? Loc[] : [p, coordenadas_x(p + vx(l), l, n - 1)...]
        trelica_especial_recta(p, h, l, n) =
          trelica_especial(coordenadas_x(p, l, n),
                           coordenadas_x(p + vxyz(l/2.0, l/2.0, -h), l, n - 1),
                           coordenadas_x(p + vy(l), l, n))
        trelica_especial_recta(u0(), 1.0, 1.0, 6)
      end,
      nothing,
      verify)

    # ── aula-06: Pantheon (CSG construction) ─────────────────────────

    run_one_test(b, slot, "aula06_corpo",
      () -> begin
        corpo(p, ri, e_corpo, e_cupula) =
          subtraction(union(cylinder(p, ri + e_corpo, ri),
                            sphere(p + vz(ri), ri + e_cupula)),
                      union(cylinder(p - vz(e_cupula), ri, ri + e_cupula),
                            sphere(p + vz(ri), ri)))
        corpo(u0(), 6.0, 1.5, 0.5)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula06_corpo_oculo",
      () -> begin
        corpo(p, ri, e_corpo, e_cupula) =
          subtraction(union(cylinder(p, ri + e_corpo, ri),
                            sphere(p + vz(ri), ri + e_cupula)),
                      union(cylinder(p - vz(e_cupula), ri, ri + e_cupula),
                            sphere(p + vz(ri), ri)))
        corpo_oculo(p, ri, e_corpo, e_cupula, r_oculo, e_oculo, h_oculo) =
          subtraction(union(corpo(p, ri, e_corpo, e_cupula),
                            cylinder(p + vz(2ri + e_cupula - h_oculo),
                                     r_oculo + e_oculo, h_oculo)),
                      cylinder(p + vz(ri), r_oculo, ri + e_cupula + h_oculo))
        corpo_oculo(u0(), 6.0, 1.5, 0.5, 1.0, 0.5, 0.4)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula06_nicho",
      () -> begin
        nicho(p, r, h) = union(cylinder(p, r, h - r), sphere(p + vz(h - r), r))
        nicho(u0(), 0.8, 4.0)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula06_corpo_oculo_nichos",
      () -> begin
        corpo(p, ri, e_corpo, e_cupula) =
          subtraction(union(cylinder(p, ri + e_corpo, ri),
                            sphere(p + vz(ri), ri + e_cupula)),
                      union(cylinder(p - vz(e_cupula), ri, ri + e_cupula),
                            sphere(p + vz(ri), ri)))
        corpo_oculo(p, ri, e_corpo, e_cupula, r_oculo, e_oculo, h_oculo) =
          subtraction(union(corpo(p, ri, e_corpo, e_cupula),
                            cylinder(p + vz(2ri + e_cupula - h_oculo),
                                     r_oculo + e_oculo, h_oculo)),
                      cylinder(p + vz(ri), r_oculo, ri + e_cupula + h_oculo))
        nicho(p, r, h) = union(cylinder(p, r, h - r), sphere(p + vz(h - r), r))
        corpo_oculo_nichos(p, ri, e_corpo, e_cupula, r_oculo, e_oculo, h_oculo,
                           r_nicho, h_nicho, n_nichos) =
          subtraction(corpo_oculo(p, ri, e_corpo, e_cupula, r_oculo, e_oculo, h_oculo),
                      union([nicho(p + vpol(ri, fi), r_nicho, h_nicho)
                             for fi in 0:2π/n_nichos:2π]))
        corpo_oculo_nichos(u0(), 6.0, 1.5, 0.5, 1.0, 0.5, 0.4, 0.6, 4.0, 4)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula06_molde_caixotao",
      () -> begin
        molde_caixotao(p, r, phi0, phi1, psi0, psi1) =
          slice(slice(slice(slice(sphere(p, r),
                                  p, vpol(1, phi0 - π/2)),
                            p, vpol(1, phi1 + π/2)),
                      p, vsph(1, (phi0 + phi1)/2, psi0 - π/2)),
                p, vsph(1, (phi0 + phi1)/2, psi1 + π/2))
        molde_caixotao(z(6.0), 5.5, 6π/5 + π/3, 6π/5 + π/2, π/3, π/2)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula06_molde_caixotao_degraus",
      () -> begin
        molde_caixotao(p, r, phi0, phi1, psi0, psi1) =
          slice(slice(slice(slice(sphere(p, r),
                                  p, vpol(1, phi0 - π/2)),
                            p, vpol(1, phi1 + π/2)),
                      p, vsph(1, (phi0 + phi1)/2, psi0 - π/2)),
                p, vsph(1, (phi0 + phi1)/2, psi1 + π/2))
        molde_caixotao_degraus(p, r, phi0, phi1, psi0, psi1, dr, da, n) =
          n == 0 ? empty_shape() :
            union(molde_caixotao(p, r, phi0, phi1, psi0, psi1),
                  molde_caixotao_degraus(p, r + dr, phi0 + da, phi1 - da,
                                         psi0 + da, psi1 - da, dr, da, n - 1))
        molde_caixotao_degraus(z(6.0), 5.5, 6π/5 + π/3, 6π/5 + π/2, π/3, π/2,
                               0.15, π/120, 3)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula06_fila_caixotoes",
      () -> begin
        molde_caixotao(p, r, phi0, phi1, psi0, psi1) =
          slice(slice(slice(slice(sphere(p, r),
                                  p, vpol(1, phi0 - π/2)),
                            p, vpol(1, phi1 + π/2)),
                      p, vsph(1, (phi0 + phi1)/2, psi0 - π/2)),
                p, vsph(1, (phi0 + phi1)/2, psi1 + π/2))
        molde_caixotao_degraus(p, r, phi0, phi1, psi0, psi1, dr, da, n) =
          n == 0 ? empty_shape() :
            union(molde_caixotao(p, r, phi0, phi1, psi0, psi1),
                  molde_caixotao_degraus(p, r + dr, phi0 + da, phi1 - da,
                                         psi0 + da, psi1 - da, dr, da, n - 1))
        fila_caixotoes(p, r, ea, n_caixotoes, psi0, psi1, dr, da, n_degraus) =
          let dphi = 2π/n_caixotoes
            union([molde_caixotao_degraus(p, r,
                                          i*dphi + ea,
                                          (i + 1)*dphi - ea,
                                          psi0, psi1, dr, da, n_degraus)
                   for i in 0:n_caixotoes - 1])
          end
        fila_caixotoes(z(6.0), 5.5, π/30, 8, π/4, π/3, 0.1, π/180, 3)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula06_panteao",
      () -> begin
        corpo(p, ri, e_corpo, e_cupula) =
          subtraction(union(cylinder(p, ri + e_corpo, ri),
                            sphere(p + vz(ri), ri + e_cupula)),
                      union(cylinder(p - vz(e_cupula), ri, ri + e_cupula),
                            sphere(p + vz(ri), ri)))
        corpo_oculo(p, ri, e_corpo, e_cupula, r_oculo, e_oculo, h_oculo) =
          subtraction(union(corpo(p, ri, e_corpo, e_cupula),
                            cylinder(p + vz(2ri + e_cupula - h_oculo),
                                     r_oculo + e_oculo, h_oculo)),
                      cylinder(p + vz(ri), r_oculo, ri + e_cupula + h_oculo))
        nicho(p, r, h) = union(cylinder(p, r, h - r), sphere(p + vz(h - r), r))
        corpo_oculo_nichos(p, ri, e_corpo, e_cupula, r_oculo, e_oculo, h_oculo,
                           r_nicho, h_nicho, n_nichos) =
          subtraction(corpo_oculo(p, ri, e_corpo, e_cupula, r_oculo, e_oculo, h_oculo),
                      union([nicho(p + vpol(ri, fi), r_nicho, h_nicho)
                             for fi in 0:2π/n_nichos:2π]))
        molde_caixotao(p, r, phi0, phi1, psi0, psi1) =
          slice(slice(slice(slice(sphere(p, r),
                                  p, vpol(1, phi0 - π/2)),
                            p, vpol(1, phi1 + π/2)),
                      p, vsph(1, (phi0 + phi1)/2, psi0 - π/2)),
                p, vsph(1, (phi0 + phi1)/2, psi1 + π/2))
        molde_caixotao_degraus(p, r, phi0, phi1, psi0, psi1, dr, da, n) =
          n == 0 ? empty_shape() :
            union(molde_caixotao(p, r, phi0, phi1, psi0, psi1),
                  molde_caixotao_degraus(p, r + dr, phi0 + da, phi1 - da,
                                         psi0 + da, psi1 - da, dr, da, n - 1))
        fila_caixotoes(p, r, ea, n_caixotoes, psi0, psi1, dr, da, n_degraus) =
          let dphi = 2π/n_caixotoes
            union([molde_caixotao_degraus(p, r,
                                          i*dphi + ea,
                                          (i + 1)*dphi - ea,
                                          psi0, psi1, dr, da, n_degraus)
                   for i in 0:n_caixotoes - 1])
          end
        conjunto_caixotoes(p, r, ea, n_filas, n_caixotoes_fila, psi0, psi1, dr, da, n_degraus) =
          n_filas == 0 ? empty_shape() :
            let dphi = 2π/n_caixotoes_fila*sin(psi0)
              union(fila_caixotoes(p, r, ea, n_caixotoes_fila,
                                   psi0 + ea, psi0 + dphi - ea,
                                   dr, da, n_degraus),
                    conjunto_caixotoes(p, r, ea, n_filas - 1, n_caixotoes_fila,
                                       psi0 + dphi, psi1, dr, da, n_degraus))
            end
        panteao(p, ri, e_corpo, e_cupula, r_oculo, e_oculo, h_oculo,
                r_nicho, h_nicho, n_nichos,
                ea, n_filas, n_caixotoes_fila, psi0, psi1, dr, da, n_degraus) =
          subtraction(corpo_oculo_nichos(p, ri, e_corpo, e_cupula,
                                         r_oculo, e_oculo, h_oculo,
                                         r_nicho, h_nicho, n_nichos),
                      conjunto_caixotoes(p + vz(ri), ri*1.01, ea,
                                         n_filas, n_caixotoes_fila,
                                         psi0, psi1, dr, da, n_degraus))
        # Scaled-down from the aula's published call
        # (panteao(u0(), 20, 4, 1, 4, 2, 1, 3, 18, 4, π/200, 5, 22, π/6, π/2, 0.1, π/300, 4))
        # to keep CSG runtime reasonable: 2 filas × 8 caixotões × 2 degraus
        # is enough to exercise the full subtraction chain.
        panteao(u0(), 6.0, 1.5, 0.5, 1.0, 0.5, 0.4, 0.6, 4.0, 4,
                π/60, 2, 8, π/6, π/2, 0.1, π/180, 2)
      end,
      nothing,
      verify)

    # ── aula-07: sinusoidal building (extruded sinusoid surfaces) ────

    run_one_test(b, slot, "aula07_laje",
      () -> begin
        sinusoide(a, omega, fi, x) = a*sin(omega*x + fi)
        pontos_sinusoide(p, a, omega, fi, x0, x1, dx) =
          x0 > x1 ? Loc[] :
            [p + vy(sinusoide(a, omega, fi, x0)),
             pontos_sinusoide(p + vx(dx), a, omega, fi, x0 + dx, x1, dx)...]
        laje(p, a, omega, fi, lx, dx, ly, lz) =
          let pontos = pontos_sinusoide(p, a, omega, fi, 0, lx, dx)
            extrusion(surface(spline(pontos),
                              line(pontos[end],
                                   p + vxy(pontos[end].x-p.x, ly),
                                   p + vxy(0, ly),
                                   pontos[1])),
                      lz)
          end
        laje(u0(), 1.0, 1.0, 0, 6.5, 0.1, 2.0, 0.2)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula07_corrimao",
      () -> begin
        sinusoide(a, omega, fi, x) = a*sin(omega*x + fi)
        pontos_sinusoide(p, a, omega, fi, x0, x1, dx) =
          x0 > x1 ? Loc[] :
            [p + vy(sinusoide(a, omega, fi, x0)),
             pontos_sinusoide(p + vx(dx), a, omega, fi, x0 + dx, x1, dx)...]
        parede_sinusoidal(p, a, omega, fi, x0, x1, dx, e, h) =
          let pts_1 = pontos_sinusoide(p, a, omega, fi, x0, x1, dx),
              pts_2 = pontos_sinusoide(p + vy(e), a, omega, fi, x0, x1, dx)
            extrusion(surface(spline(pts_1),
                              spline([pts_1[end], pts_2[end]]),
                              spline(reverse(pts_2)),
                              spline([pts_2[1], pts_1[1]])),
                      h)
          end
        corrimao(p, a, omega, fi, lx, dx, l_corrimao, a_corrimao) =
          parede_sinusoidal(p, a, omega, fi, 0, lx, dx, l_corrimao, a_corrimao)
        corrimao(u0(), 1.0, 1.0, 0, 6.5, 0.1, 0.06, 0.02)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula07_prumos",
      () -> begin
        sinusoide(a, omega, fi, x) = a*sin(omega*x + fi)
        pontos_sinusoide(p, a, omega, fi, x0, x1, dx) =
          x0 > x1 ? Loc[] :
            [p + vy(sinusoide(a, omega, fi, x0)),
             pontos_sinusoide(p + vx(dx), a, omega, fi, x0 + dx, x1, dx)...]
        prumos(p, a, omega, fi, lx, dx, altura, raio) =
          for ponto in pontos_sinusoide(p, a, omega, fi, 0, lx, dx)
            cylinder(ponto, raio, ponto + vxyz(0, 0, altura))
          end
        prumos(u0(), 1.0, 1.0, 0, 6.5, 0.5, 1.0, 0.02)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula07_guarda",
      () -> begin
        sinusoide(a, omega, fi, x) = a*sin(omega*x + fi)
        pontos_sinusoide(p, a, omega, fi, x0, x1, dx) =
          x0 > x1 ? Loc[] :
            [p + vy(sinusoide(a, omega, fi, x0)),
             pontos_sinusoide(p + vx(dx), a, omega, fi, x0 + dx, x1, dx)...]
        parede_sinusoidal(p, a, omega, fi, x0, x1, dx, e, h) =
          let pts_1 = pontos_sinusoide(p, a, omega, fi, x0, x1, dx),
              pts_2 = pontos_sinusoide(p + vy(e), a, omega, fi, x0, x1, dx)
            extrusion(surface(spline(pts_1),
                              spline([pts_1[end], pts_2[end]]),
                              spline(reverse(pts_2)),
                              spline([pts_2[1], pts_1[1]])),
                      h)
          end
        corrimao(p, a, omega, fi, lx, dx, l_corrimao, a_corrimao) =
          parede_sinusoidal(p, a, omega, fi, 0, lx, dx, l_corrimao, a_corrimao)
        prumos(p, a, omega, fi, lx, dx, altura, raio) =
          for ponto in pontos_sinusoide(p, a, omega, fi, 0, lx, dx)
            cylinder(ponto, raio, ponto + vxyz(0, 0, altura))
          end
        guarda(p, a, omega, fi, lx, dx, a_guarda, l_corrimao, a_corrimao, d_prumos) =
          begin
            corrimao(p + vxyz(0, l_corrimao/-2.0, a_guarda),
                     a, omega, fi, lx, d_prumos, l_corrimao, a_corrimao)
            prumos(p, a, omega, fi, lx, d_prumos, a_guarda, l_corrimao/3.0)
          end
        guarda(u0(), 1.0, 1.0, 0, 6.5, 0.5, 1.0, 0.06, 0.02, 0.4)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula07_piso",
      () -> begin
        sinusoide(a, omega, fi, x) = a*sin(omega*x + fi)
        pontos_sinusoide(p, a, omega, fi, x0, x1, dx) =
          x0 > x1 ? Loc[] :
            [p + vy(sinusoide(a, omega, fi, x0)),
             pontos_sinusoide(p + vx(dx), a, omega, fi, x0 + dx, x1, dx)...]
        laje(p, a, omega, fi, lx, dx, ly, lz) =
          let pontos = pontos_sinusoide(p, a, omega, fi, 0, lx, dx)
            extrusion(surface(spline(pontos),
                              line(pontos[end],
                                   p + vxy(pontos[end].x-p.x, ly),
                                   p + vxy(0, ly),
                                   pontos[1])),
                      lz)
          end
        parede_sinusoidal(p, a, omega, fi, x0, x1, dx, e, h) =
          let pts_1 = pontos_sinusoide(p, a, omega, fi, x0, x1, dx),
              pts_2 = pontos_sinusoide(p + vy(e), a, omega, fi, x0, x1, dx)
            extrusion(surface(spline(pts_1),
                              spline([pts_1[end], pts_2[end]]),
                              spline(reverse(pts_2)),
                              spline([pts_2[1], pts_1[1]])),
                      h)
          end
        corrimao(p, a, omega, fi, lx, dx, l_corrimao, a_corrimao) =
          parede_sinusoidal(p, a, omega, fi, 0, lx, dx, l_corrimao, a_corrimao)
        prumos(p, a, omega, fi, lx, dx, altura, raio) =
          for ponto in pontos_sinusoide(p, a, omega, fi, 0, lx, dx)
            cylinder(ponto, raio, ponto + vxyz(0, 0, altura))
          end
        guarda(p, a, omega, fi, lx, dx, a_guarda, l_corrimao, a_corrimao, d_prumos) =
          begin
            corrimao(p + vxyz(0, l_corrimao/-2.0, a_guarda),
                     a, omega, fi, lx, d_prumos, l_corrimao, a_corrimao)
            prumos(p, a, omega, fi, lx, d_prumos, a_guarda, l_corrimao/3.0)
          end
        piso(p, a, omega, fi, lx, dx, ly, a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos) =
          begin
            laje(p, a, omega, fi, lx, d_prumos, ly, a_laje)
            guarda(p + vxyz(0, l_corrimao, a_laje),
                   a, omega, fi, lx, dx, a_guarda, l_corrimao, a_corrimao, d_prumos)
          end
        piso(u0(), 1.0, 1.0, 0, 6.5, 0.5, 2.0, 0.2, 1.0, 0.06, 0.02, 0.4)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula07_predio_v1",
      () -> begin
        sinusoide(a, omega, fi, x) = a*sin(omega*x + fi)
        pontos_sinusoide(p, a, omega, fi, x0, x1, dx) =
          x0 > x1 ? Loc[] :
            [p + vy(sinusoide(a, omega, fi, x0)),
             pontos_sinusoide(p + vx(dx), a, omega, fi, x0 + dx, x1, dx)...]
        laje(p, a, omega, fi, lx, dx, ly, lz) =
          let pontos = pontos_sinusoide(p, a, omega, fi, 0, lx, dx)
            extrusion(surface(spline(pontos),
                              line(pontos[end],
                                   p + vxy(pontos[end].x-p.x, ly),
                                   p + vxy(0, ly),
                                   pontos[1])),
                      lz)
          end
        parede_sinusoidal(p, a, omega, fi, x0, x1, dx, e, h) =
          let pts_1 = pontos_sinusoide(p, a, omega, fi, x0, x1, dx),
              pts_2 = pontos_sinusoide(p + vy(e), a, omega, fi, x0, x1, dx)
            extrusion(surface(spline(pts_1),
                              spline([pts_1[end], pts_2[end]]),
                              spline(reverse(pts_2)),
                              spline([pts_2[1], pts_1[1]])),
                      h)
          end
        corrimao(p, a, omega, fi, lx, dx, l_corrimao, a_corrimao) =
          parede_sinusoidal(p, a, omega, fi, 0, lx, dx, l_corrimao, a_corrimao)
        prumos(p, a, omega, fi, lx, dx, altura, raio) =
          for ponto in pontos_sinusoide(p, a, omega, fi, 0, lx, dx)
            cylinder(ponto, raio, ponto + vxyz(0, 0, altura))
          end
        guarda(p, a, omega, fi, lx, dx, a_guarda, l_corrimao, a_corrimao, d_prumos) =
          begin
            corrimao(p + vxyz(0, l_corrimao/-2.0, a_guarda),
                     a, omega, fi, lx, d_prumos, l_corrimao, a_corrimao)
            prumos(p, a, omega, fi, lx, d_prumos, a_guarda, l_corrimao/3.0)
          end
        piso(p, a, omega, fi, lx, dx, ly, a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos) =
          begin
            laje(p, a, omega, fi, lx, d_prumos, ly, a_laje)
            guarda(p + vxyz(0, l_corrimao, a_laje),
                   a, omega, fi, lx, dx, a_guarda, l_corrimao, a_corrimao, d_prumos)
          end
        predio(p, a, omega, fi, lx, dx, ly,
               a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos, a_andar, n_andares) =
          n_andares == 0 ? nothing :
            begin
              piso(p, a, omega, fi, lx, dx, ly,
                   a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos)
              predio(p + vxyz(0, 0, a_andar), a, omega, fi, lx, dx, ly,
                     a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos, a_andar, n_andares - 1)
            end
        # Original aula call: predio(xy(0,0), 1.0, 1.0, 0, 60, 0.5, 20, 0.2, 1, 0.06, 0.02, 0.4, 4, 10).
        # Scaled to lx=10, ly=4, n_andares=3 to keep per-piso CSG tractable.
        predio(u0(), 1.0, 1.0, 0, 10.0, 0.5, 4.0, 0.2, 1.0, 0.06, 0.02, 0.4, 4.0, 3)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula07_predio_v2",
      () -> begin
        sinusoide(a, omega, fi, x) = a*sin(omega*x + fi)
        pontos_sinusoide(p, a, omega, fi, x0, x1, dx) =
          x0 > x1 ? Loc[] :
            [p + vy(sinusoide(a, omega, fi, x0)),
             pontos_sinusoide(p + vx(dx), a, omega, fi, x0 + dx, x1, dx)...]
        laje(p, a, omega, fi, lx, dx, ly, lz) =
          let pontos = pontos_sinusoide(p, a, omega, fi, 0, lx, dx)
            extrusion(surface(spline(pontos),
                              line(pontos[end],
                                   p + vxy(pontos[end].x-p.x, ly),
                                   p + vxy(0, ly),
                                   pontos[1])),
                      lz)
          end
        parede_sinusoidal(p, a, omega, fi, x0, x1, dx, e, h) =
          let pts_1 = pontos_sinusoide(p, a, omega, fi, x0, x1, dx),
              pts_2 = pontos_sinusoide(p + vy(e), a, omega, fi, x0, x1, dx)
            extrusion(surface(spline(pts_1),
                              spline([pts_1[end], pts_2[end]]),
                              spline(reverse(pts_2)),
                              spline([pts_2[1], pts_1[1]])),
                      h)
          end
        corrimao(p, a, omega, fi, lx, dx, l_corrimao, a_corrimao) =
          parede_sinusoidal(p, a, omega, fi, 0, lx, dx, l_corrimao, a_corrimao)
        prumos(p, a, omega, fi, lx, dx, altura, raio) =
          for ponto in pontos_sinusoide(p, a, omega, fi, 0, lx, dx)
            cylinder(ponto, raio, ponto + vxyz(0, 0, altura))
          end
        guarda(p, a, omega, fi, lx, dx, a_guarda, l_corrimao, a_corrimao, d_prumos) =
          begin
            corrimao(p + vxyz(0, l_corrimao/-2.0, a_guarda),
                     a, omega, fi, lx, d_prumos, l_corrimao, a_corrimao)
            prumos(p, a, omega, fi, lx, d_prumos, a_guarda, l_corrimao/3.0)
          end
        piso(p, a, omega, fi, lx, dx, ly, a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos) =
          begin
            laje(p, a, omega, fi, lx, d_prumos, ly, a_laje)
            guarda(p + vxyz(0, l_corrimao, a_laje),
                   a, omega, fi, lx, dx, a_guarda, l_corrimao, a_corrimao, d_prumos)
          end
        # Per-floor phase increment dfi — generates twisted facade.
        predio(p, a, omega, fi, lx, dx, ly,
               a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos, a_andar, n_andares, dfi) =
          n_andares == 0 ? nothing :
            begin
              piso(p, a, omega, fi, lx, dx, ly,
                   a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos)
              predio(p + vxyz(0, 0, a_andar), a, omega, fi + dfi, lx, dx, ly,
                     a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos, a_andar, n_andares - 1, dfi)
            end
        predio(u0(), 1.0, 1.0, 0, 10.0, 0.5, 4.0, 0.2, 1.0, 0.06, 0.02, 0.4, 4.0, 3, π/4)
      end,
      nothing,
      verify)

    # ── aula-08: revolved profiles (sinusoidal tube, barrel) ─────────

    run_one_test(b, slot, "aula08_tubo_sinusoidal",
      () -> begin
        sinusoide(a, omega, fi, x) = a*sin(omega*x + fi)
        pontos_sinusoide(p, a, omega, fi, x0, x1, dx) =
          x0 > x1 ? Loc[] :
            [p + vxy(x0, sinusoide(a, omega, fi, x0)),
             pontos_sinusoide(p, a, omega, fi, x0 + dx, x1, dx)...]
        tubo_sinusoidal(p, r, a, omega, fi, lx, dx) =
          revolve(spline(pontos_sinusoide(p + vxy(0, -r), a, omega, fi, 0, lx, dx)),
                  p, vx(1))
        tubo_sinusoidal(u0(), 5.0, 1.0, 1.0, 0.0, 20.0, 0.4)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula08_barril",
      () -> begin
        perfil_barril(p, r0, r1, h, e) =
          surface(line(p + vcyl(r0, 0, 0),
                       p,
                       p + vcyl(0, 0, e),
                       p + vcyl(r0 - e, 0, e)),
                  spline(p + vcyl(r0 - e, 0, e),
                         p + vcyl(r1 - e, 0, h/2),
                         p + vcyl(r0 - e, 0, h - e)),
                  line(p + vcyl(r0 - e, 0, h - e),
                       p + vcyl(r0 - e, 0, h),
                       p + vcyl(r0, 0, h)),
                  spline(p + vcyl(r0, 0, h),
                         p + vcyl(r1, 0, h/2),
                         p + vcyl(r0, 0, 0)))
        barril(p, r0, r1, h, e) = revolve(perfil_barril(p, r0, r1, h, e), p)
        barril(u0(), 1.0, 1.3, 4.0, 0.1)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula08_perfil_barril",
      () -> begin
        perfil_barril(p, r0, r1, h, e) =
          surface(line(p + vcyl(r0, 0, 0),
                       p,
                       p + vcyl(0, 0, e),
                       p + vcyl(r0 - e, 0, e)),
                  spline(p + vcyl(r0 - e, 0, e),
                         p + vcyl(r1 - e, 0, h/2),
                         p + vcyl(r0 - e, 0, h - e)),
                  line(p + vcyl(r0 - e, 0, h - e),
                       p + vcyl(r0 - e, 0, h),
                       p + vcyl(r0, 0, h)),
                  spline(p + vcyl(r0, 0, h),
                         p + vcyl(r1, 0, h/2),
                         p + vcyl(r0, 0, 0)))
        perfil_barril(u0(), 1.0, 1.3, 4.0, 0.1)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula08_tabua_barril",
      () -> begin
        perfil_barril(p, r0, r1, h, e) =
          surface(line(p + vcyl(r0, 0, 0),
                       p,
                       p + vcyl(0, 0, e),
                       p + vcyl(r0 - e, 0, e)),
                  spline(p + vcyl(r0 - e, 0, e),
                         p + vcyl(r1 - e, 0, h/2),
                         p + vcyl(r0 - e, 0, h - e)),
                  line(p + vcyl(r0 - e, 0, h - e),
                       p + vcyl(r0 - e, 0, h),
                       p + vcyl(r0, 0, h)),
                  spline(p + vcyl(r0, 0, h),
                         p + vcyl(r1, 0, h/2),
                         p + vcyl(r0, 0, 0)))
        tabua_barril(p, r0, r1, h, e, alfa, d_alfa) =
          revolve(perfil_barril(p, r0, r1, h, e), p, vz(1), alfa, d_alfa)
        tabua_barril(u0(), 1.0, 1.3, 4.0, 0.1, 0.0, π/3)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula08_tabuas_barril",
      () -> begin
        perfil_barril(p, r0, r1, h, e) =
          surface(line(p + vcyl(r0, 0, 0),
                       p,
                       p + vcyl(0, 0, e),
                       p + vcyl(r0 - e, 0, e)),
                  spline(p + vcyl(r0 - e, 0, e),
                         p + vcyl(r1 - e, 0, h/2),
                         p + vcyl(r0 - e, 0, h - e)),
                  line(p + vcyl(r0 - e, 0, h - e),
                       p + vcyl(r0 - e, 0, h),
                       p + vcyl(r0, 0, h)),
                  spline(p + vcyl(r0, 0, h),
                         p + vcyl(r1, 0, h/2),
                         p + vcyl(r0, 0, 0)))
        tabua_barril(p, r0, r1, h, e, alfa, d_alfa) =
          revolve(perfil_barril(p, r0, r1, h, e), p, vz(1), alfa, d_alfa)
        tabuas_barril(p, r0, r1, h, e, n, s) =
          let delta_alfa = 2π/n
            for alfa in division(0, 2π, n, false)
              tabua_barril(p, r0, r1, h, e, alfa, delta_alfa - s)
            end
          end
        tabuas_barril(u0(), 1.0, 1.3, 4.0, 0.1, 6, 0.05)
      end,
      nothing,
      verify)

    # ── aula-09: superelipse-walled tanks ────────────────────────────

    run_one_test(b, slot, "aula09_curva_superelipse",
      () -> begin
        superelipse(p, a, b, n, t) =
          p + vxy(a*(cos(t)^2)^(1/n)*sign(cos(t)),
                  b*(sin(t)^2)^(1/n)*sign(sin(t)))
        pontos_superelipse(p, a, b, n, pts) =
          [superelipse(p, a, b, n, t) for t in division(-pi, pi, pts, false)]
        curva_superelipse(p, a, b, n, pts) =
          closed_spline(pontos_superelipse(p, a, b, n, pts))
        curva_superelipse(xy(0, 0), 2.0, 4.0, 2.5, 50)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula09_tanque_superelipse",
      () -> begin
        superelipse(p, a, b, n, t) =
          p + vxy(a*(cos(t)^2)^(1/n)*sign(cos(t)),
                  b*(sin(t)^2)^(1/n)*sign(sin(t)))
        pontos_superelipse(p, a, b, n, pts) =
          [superelipse(p, a, b, n, t) for t in division(-pi, pi, pts, false)]
        curva_superelipse(p, a, b, n, pts) =
          closed_spline(pontos_superelipse(p, a, b, n, pts))
        parede_curva(espessura, altura, curva) =
          let seccao = surface_rectangle(u0(), xy(espessura, altura))
            sweep(curva, seccao)
          end
        tanque_superelipse(p, espessura, altura, a, b, n, pts) =
          parede_curva(espessura, altura,
                       curva_superelipse(p, a, b, n, pts))
        tanque_superelipse(u0(), 0.4, 1.0, 5.0, 7.0, 2.5, 40)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula09_tanque_circular",
      () -> begin
        parede_curva(espessura, altura, curva) =
          let seccao = surface_rectangle(u0(), xy(espessura, altura))
            sweep(curva, seccao)
          end
        tanque_circular(p, raio, espessura, altura) =
          parede_curva(espessura, altura, circle(p, raio))
        tanque_circular(u0(), 3.0, 0.2, 0.8)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula09_tanques_circulares",
      () -> begin
        superelipse(p, a, b, n, t) =
          p + vxy(a*(cos(t)^2)^(1/n)*sign(cos(t)),
                  b*(sin(t)^2)^(1/n)*sign(sin(t)))
        parede_curva(espessura, altura, curva) =
          let seccao = surface_rectangle(u0(), xy(espessura, altura))
            sweep(curva, seccao)
          end
        tanque_circular(p, raio, espessura, altura) =
          parede_curva(espessura, altura, circle(p, raio))
        tanques_circulares(p, raio, espessura, altura, a, b, n, pts) =
          for t in division(-π, π, pts, false)
            tanque_circular(superelipse(p, a, b, n, t), raio, espessura, altura)
          end
        tanques_circulares(u0(), 0.4, 0.1, 0.6, 5.0, 6.0, 2.5, 12)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula09_tanque_sergels",
      () -> begin
        superelipse(p, a, b, n, t) =
          p + vxy(a*(cos(t)^2)^(1/n)*sign(cos(t)),
                  b*(sin(t)^2)^(1/n)*sign(sin(t)))
        pontos_superelipse(p, a, b, n, pts) =
          [superelipse(p, a, b, n, t) for t in division(-π, π, pts, false)]
        curva_superelipse(p, a, b, n, pts) =
          closed_spline(pontos_superelipse(p, a, b, n, pts))
        parede_curva(espessura, altura, curva) =
          let seccao = surface_rectangle(u0(), xy(espessura, altura))
            sweep(curva, seccao)
          end
        tanque_superelipse(p, espessura, altura, a, b, n, pts) =
          parede_curva(espessura, altura, curva_superelipse(p, a, b, n, pts))
        tanque_circular(p, raio, espessura, altura) =
          parede_curva(espessura, altura, circle(p, raio))
        tanques_circulares(p, raio, espessura, altura, a, b, n, pts) =
          for t in division(-π, π, pts, false)
            tanque_circular(superelipse(p, a, b, n, t), raio, espessura, altura)
          end
        tanque_sergels(p) = begin
          tanque_superelipse(p, 0.4, 1.0, 6.0, 7.0, 2.5, 40)
          tanques_circulares(p, 0.3, 0.1, 0.5, 5.0, 6.0, 2.5, 12)
        end
        tanque_sergels(u0())
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula09_tanque_sergels_circular",
      () -> begin
        superelipse(p, a, b, n, t) =
          p + vxy(a*(cos(t)^2)^(1/n)*sign(cos(t)),
                  b*(sin(t)^2)^(1/n)*sign(sin(t)))
        pontos_superelipse(p, a, b, n, pts) =
          [superelipse(p, a, b, n, t) for t in division(-π, π, pts, false)]
        curva_superelipse(p, a, b, n, pts) =
          closed_spline(pontos_superelipse(p, a, b, n, pts))
        parede_curva(espessura, altura, curva) =
          let seccao = surface_rectangle(u0(), xy(espessura, altura))
            sweep(curva, seccao)
          end
        tanque_superelipse(p, espessura, altura, a, b, n, pts) =
          parede_curva(espessura, altura, curva_superelipse(p, a, b, n, pts))
        tanque_circular(p, raio, espessura, altura) =
          parede_curva(espessura, altura, circle(p, raio))
        tanques_circulares(p, raio, espessura, altura, a, b, n, pts) =
          for t in division(-π, π, pts, false)
            tanque_circular(superelipse(p, a, b, n, t), raio, espessura, altura)
          end
        tanque_sergels(p) =
          begin
            tanque_superelipse(p, 0.4, 1.0, 7.0, 7.0, 2, 80)
            tanques_circulares(p, 0.6, 0.2, 0.5, 6.0, 6.0, 2, 24)
            tanques_circulares(p, 0.5, 0.2, 0.5, 5.0, 5.0, 2, 24)
          end
        tanque_sergels(u0())
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula09_tanque_sergels_alternative",
      () -> begin
        superelipse(p, a, b, n, t) =
          p + vxy(a*(cos(t)^2)^(1/n)*sign(cos(t)),
                  b*(sin(t)^2)^(1/n)*sign(sin(t)))
        pontos_superelipse(p, a, b, n, pts) =
          [superelipse(p, a, b, n, t) for t in division(-π, π, pts, false)]
        curva_superelipse(p, a, b, n, pts) =
          closed_spline(pontos_superelipse(p, a, b, n, pts))
        parede_curva(espessura, altura, curva) =
          let seccao = surface_rectangle(u0(), xy(espessura, altura))
            sweep(curva, seccao)
          end
        tanque_superelipse(p, espessura, altura, a, b, n, pts) =
          parede_curva(espessura, altura, curva_superelipse(p, a, b, n, pts))
        tanque_circular(p, raio, espessura, altura) =
          parede_curva(espessura, altura, circle(p, raio))
        tanques_circulares(p, raio, espessura, altura, a, b, n, pts) =
          for t in division(-π, π, pts, false)
            tanque_circular(superelipse(p, a, b, n, t), raio, espessura, altura)
          end
        tanque_sergels(p) =
          begin
            tanque_superelipse(p, 0.4, 1.0, 20, 22, 1.3, 100)
            tanques_circulares(p, 1.0, 0.2, 0.8, 18, 20, 1.3, 20)
            tanques_circulares(p, 0.8, 0.2, 0.8, 15, 17, 1.3, 20)
            tanques_circulares(p, 0.6, 0.2, 0.8, 12, 14, 1.3, 20)
            tanques_circulares(p, 0.4, 0.2, 0.8, 10, 12, 1.3, 20)
          end
        tanque_sergels(u0())
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula09_tanque_sergels_alternative_alternative",
      () -> begin
        superelipse(p, a, b, n, t) =
          p + vxy(a*(cos(t)^2)^(1/n)*sign(cos(t)),
                  b*(sin(t)^2)^(1/n)*sign(sin(t)))
        pontos_superelipse(p, a, b, n, pts) =
          [superelipse(p, a, b, n, t) for t in division(-π, π, pts, false)]
        curva_superelipse(p, a, b, n, pts) =
          closed_spline(pontos_superelipse(p, a, b, n, pts))
        parede_curva(espessura, altura, curva) =
          let seccao = surface_rectangle(u0(), xy(espessura, altura))
            sweep(curva, seccao)
          end
        tanque_superelipse(p, espessura, altura, a, b, n, pts) =
          for s in division(-pi, pi, 4, false)
            parede_curva(espessura, 
                         altura,
                         spline([superelipse(p, a, b, n, t)
                                 for t in division(s, 
                                                   s + pi/2,
                                                   floor(Int, pts/4),
                                                   true)]))
          end
        tanque_circular(p, raio, espessura, altura) =
          parede_curva(espessura, altura, circle(p, raio))
        tanques_circulares(p, raio, espessura, altura, a, b, n, pts) =
          for t in division(-π, π, pts, false)
            tanque_circular(superelipse(p, a, b, n, t), raio, espessura, altura)
          end
        tanque_sergels(p) =
          begin
            tanque_superelipse(p, 0.4, 1.0, 22, 22, 0.7, 100)
            tanques_circulares(p, 1.0, 0.2, 0.8, 17, 17, 0.7, 8)
            tanques_circulares(p, 1.0, 0.2, 0.8, 12, 12, 0.7, 12)
          end
        tanque_sergels(u0())
      end,
      nothing,
      verify)

    # ── aula-10: parametric surfaces (ellipsoid, torus, dini) ────────

    run_one_test(b, slot, "aula10_elipsoide_open",
      () -> begin
        ponto_elipsoide(p, a, b, c, fi, psi) =
          p + vxyz(a*sin(psi)*cos(fi),
                   b*sin(psi)*sin(fi),
                   c*cos(psi))
        pontos_elipsoide(p, a, b, c, n) =
          map_division((fi, psi) -> ponto_elipsoide(p, a, b, c, fi, psi),
                       -pi/2, pi/2, n,
                       -pi, pi, n)
        elipsoide(p, a, b, c, n) = surface_grid(pontos_elipsoide(p, a, b, c, n))
        elipsoide(xy(0, 0), 2, 3, 4, 30)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula10_elipsoide_closed",
      () -> begin
        ponto_elipsoide(p, a, b, c, fi, psi) =
          p + vxyz(a*sin(psi)*cos(fi),
                   b*sin(psi)*sin(fi),
                   c*cos(psi))
        elipsoide(p, a, b, c, n) =
          surface_grid(
            map_division((fi, psi) -> ponto_elipsoide(p, a, b, c, fi, psi),
                         0, 2*pi, n, false,
                         0, pi, n, true),
            true, false)
        elipsoide(xy(0, 0), 2, 3, 4, 30)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula10_toro",
      () -> begin
        toro(p, r0, r1, m, n) =
          surface_grid(
            map_division((fi, psi) -> p + vpol(r0, fi) + vsph(r1, fi, psi),
                         0, 2pi, m, false,
                         0, 2pi, n, false),
            true, true)
        toro(u0(), 10.0, 2.0, 40, 16)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula10_meio_toro",
      () -> begin
        meio_toro(p, r0, r1, m, n) =
          surface_grid(
            map_division((fi, psi) -> p + vpol(r0, fi) + vsph(r1, fi, psi),
                         0, 2pi, m, false,
                         pi/2, 3*pi/2, n, true),
            true, false)
        meio_toro(u0(), 10.0, 2.0, 40, 16)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula10_dini",
      () -> begin
        dini(p, a, u0_, u1, n, v0, v1, m) =
          map_division((u, v) -> p + vxyz(cos(u)*sin(v),
                                           sin(u)*sin(v),
                                           cos(v) + log(tan(v/2)) + a*u),
                       u0_, u1, m,
                       v0, v1, n)
        surface_grid(dini(xyz(0, 0, 0), 0.2, 0, 6*pi, 60, 0.1, pi*0.5, 60))
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula10_maca",
      () -> begin
        maca(p) =
          surface_grid(
            map_division((u, v) -> p + vxyz(cos(u)*(4 + 3.8*cos(v)),
                                             sin(u)*(4 + 3.8*cos(v)),
                                             (cos(v) + sin(v) - 1)*(1 + sin(v))*log(1 - π*v/10) + 7.5*sin(v)),
                         0, 2π, 30,
                         -π, π, 30))
        maca(u0())
      end,
      nothing,
      verify)

    # ── aula-11: quad-iterator decorations on parametric surfaces ────

    run_one_test(b, slot, "aula11_superficie_esferas",
      () -> begin
        itera_quadrangulos(f, ptss) =
          [[f(p0, p1, p2, p3)
            for (p0, p1, p2, p3)
            in zip(pts0[1:end-1], pts1[1:end-1], pts1[2:end], pts0[2:end])]
           for (pts0, pts1)
           in zip(ptss[1:end-1], ptss[2:end])]
        sin_u_mul_v(n) =
          map_division((u, v) -> xyz(u, v, 0.4*sin(u*v)),
                       -pi, pi, n,
                       -pi, pi, n)
        esfera_quadrangulo(p0, p1, p2, p3) =
          let raio = min(distance(p0, p1), distance(p1, p2),
                         distance(p2, p3), distance(p3, p0))/2
            sphere(quad_center(p0, p1, p2, p3), raio)
          end
        superficie_esferas(ptss) = itera_quadrangulos(esfera_quadrangulo, ptss)
        superficie_esferas(sin_u_mul_v(20))
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula11_superficie_cones",
      () -> begin
        itera_quadrangulos(f, ptss) =
          [[f(p0, p1, p2, p3)
            for (p0, p1, p2, p3)
            in zip(pts0[1:end-1], pts1[1:end-1], pts1[2:end], pts0[2:end])]
           for (pts0, pts1)
           in zip(ptss[1:end-1], ptss[2:end])]
        sin_u_mul_v(n) =
          map_division((u, v) -> xyz(u, v, 0.4*sin(u*v)),
                       -pi, pi, n,
                       -pi, pi, n)
        cone_quadrangulo(p0, p1, p2, p3) =
          let p = quad_center(p0, p1, p2, p3)
              n = quad_normal(p0, p1, p2, p3)
              d = min(distance(p0, p1), distance(p0, p3))
            cone(p, d/2, p + n*d)
          end
        superficie_cones(ptss) = itera_quadrangulos(cone_quadrangulo, ptss)
        superficie_cones(sin_u_mul_v(20))
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula11_malha",
      () -> begin
        itera_quadrangulos(f, ptss) =
          [[f(p0, p1, p2, p3)
            for (p0, p1, p2, p3)
            in zip(pts0[1:end-1], pts1[1:end-1], pts1[2:end], pts0[2:end])]
           for (pts0, pts1) in zip(ptss[1:end-1], ptss[2:end])]
        toro(p, r0, r1, m, n) =
          [[p + vcyl(r0 + r1*cos(v), u, r1*sin(v))
            for v in division(0, 2π, n)]
           for u in division(0, 2π, m)]
        cruz_quadrangulo(p0, p1, p2, p3) =
          let raio = min(distance(p0, p1), distance(p1, p2),
                         distance(p2, p3), distance(p3, p0))/6
            union(cylinder(p0, raio, p2),
                  cylinder(p1, raio, p3))
          end
        malha(ptss) = itera_quadrangulos(cruz_quadrangulo, ptss)
        malha(toro(u0(), 10.0, 4.0, 24, 12))
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula11_superficie_piramides",
      () -> begin
        itera_quadrangulos(f, ptss) =
          [[f(p0, p1, p2, p3)
            for (p0, p1, p2, p3)
            in zip(pts0[1:end-1], pts1[1:end-1], pts1[2:end], pts0[2:end])]
           for (pts0, pts1)
           in zip(ptss[1:end-1], ptss[2:end])]
        toro(p, r0, r1, m, n) =
          [[p + vcyl(r0 + r1*cos(v), u, r1*sin(v))
            for v in division(0, 2*pi, n)]
           for u in division(0, 2*pi, m)]
        piramide_quadrangulo(p0, p1, p2, p3) =
          let p = quad_center(p0, p1, p2, p3)
              n = quad_normal(p0, p1, p2, p3)
              d = min(distance(p0, p1), distance(p0, p3))
            loft([polygon(p0, p1, p2, p3), point(p + n*d)])
          end
        superficie_piramides(ptss) = itera_quadrangulos(piramide_quadrangulo, ptss)
        superficie_piramides(toro(u0(), 10, 4, 16, 8))
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula11_superficie_cilindros",
      () -> begin
        itera_quadrangulos(f, ptss) =
          [[f(p0, p1, p2, p3)
            for (p0, p1, p2, p3)
            in zip(pts0[1:end-1], pts1[1:end-1], pts1[2:end], pts0[2:end])]
           for (pts0, pts1)
           in zip(ptss[1:end-1], ptss[2:end])]
        sin_u_mul_v(n) =
          map_division((u, v) -> xyz(u, v, 0.4*sin(u*v)),
                       -pi, pi, n,
                       -pi, pi, n)
        cilindro_quadrangulo(p0, p1, p2, p3, min_z) =
          let raio = min(distance(p0, p1), distance(p1, p2),
                         distance(p2, p3), distance(p3, p0))/2
              topo = quad_center(p0, p1, p2, p3)
            cylinder(xyz(topo.x, topo.y, min_z), raio, topo)
          end
        superficie_cilindros(ptss) =
          let min_z = reduce(min,
                             map(pts -> reduce(min, map(cz, pts)), ptss))
            itera_quadrangulos(
              (p0, p1, p2, p3) -> cilindro_quadrangulo(p0, p1, p2, p3, min_z),
              ptss)
          end
        superficie_cilindros(sin_u_mul_v(20))
      end,
      nothing,
      verify)

    run_one_test(b, slot, "aula11_superficie_borbulhas",
      () -> begin
        sin_u_mul_v(n) =
          map_division((u, v) -> xyz(u, v, 0.4*sin(u*v)),
                       -π, π, n,
                       -π, π, n)
        itera_quadrangulos(f, ptss) =
          [[f(p0, p1, p2, p3)
            for (p0, p1, p2, p3)
            in zip(pts0[1:end-1], pts1[1:end-1], pts1[2:end], pts0[2:end])]
           for (pts0, pts1)
           in zip(ptss[1:end-1], ptss[2:end])]
        borbulha(p0, p1, p2, p3) =
          let p = quad_center(p0, p1, p2, p3),
              n = quad_normal(p0, p1, p2, p3),
              d = min(distance(p0, p1), distance(p0, p3)),
              c = loc_from_o_vz(p, n)
            loft([polygon(p0, p1, p2, p3),
                  circle(c + vz(d/10.0), d/3.0),
                  circle(c + vz(d/3.0), d/4.0),
                  circle(c, d/5.0)])
          end
        superficie_borbulhas(ptss) = itera_quadrangulos(borbulha, ptss)
        superficie_borbulhas(sin_u_mul_v(12))
      end,
      nothing,
      verify)

    # ── map15-02a: losango ───────────────────────────────────────────
    run_one_test(b, slot, "map15_02a_losango",
      () -> begin
        losango(p, l, h) =
          let p1 = p + vx(l/2)
              p2 = p + vy(h/2)
              p3 = p - vx(l/2)
              p4 = p - vy(h/2)
            polygon(p1, p2, p3, p4)
          end
        losango(u0(), 4.0, 8.0)
      end,
      nothing,
      verify)

    # ── map15-03b: baravelle spiral ──────────────────────────────────
    run_one_test(b, slot, "map15_03b_baravelle_spiral",
      () -> begin
        baravelle_spiral(p, l, n, α) =
          n == 0 ?
            nothing :
            let p1 = p + vpol(l, α),
                p2 = p1 + vpol(l, α + π/2),
                d = sqrt(2)*l/2,
                p3 = p + vpol(d, α + π/4)
              polygon(p, p1, p2)
              baravelle_spiral(p3, d, n - 1, α + π/4)
            end
        baravelle_spiral(u0(), 1.0, 10, 0.0)
      end,
      nothing,
      verify)

    # ── map15-05a: anel sinusoidal (sweep of closed-spline ring) ─────
    run_one_test(b, slot, "map15_05a_anel_sinusoidal",
      () -> begin
        anel_sinusoidal(p, re, ri, a, omega, n) =
          sweep(
            closed_spline(
              map_division(t -> cyl(re, t, a*sin(omega*t)), 0, 2*pi, n, false)),
            surface_circle(x(0), ri))
        anel_sinusoidal(u0(), 5.0, 0.3, 0.5, 4.0, 60)
      end,
      nothing,
      verify)

    # ── map15-06a: bottle stopper (revolved sawtooth profile) ────────
    run_one_test(b, slot, "map15_06a_rolha",
      () -> begin
        pts_rolha(p, a, c1, c2, n) =
          n == 0 ? [p] :
            [p, p + vxy(c1, -a), p + vxy(c1 + c2, -a),
             pts_rolha(p + vx(c1 + c2), a, c1, c2, n - 1)...]
        seccao_rolha(p, a, c1, c2, n, r) =
          polygon([p,
                   pts_rolha(p - vy(r), a, c1, c2, n)...,
                   p + vx((c1 + c2)*n)])
        rolha(p, a, c1, c2, n, r) =
          revolve(seccao_rolha(p, a, c1, c2, n, r), p, vx())
        rolha(u0(), 2.0, 2.0, 5.0, 5, 10.0)
      end,
      nothing,
      verify)

    # ── map15-06b: zigzag slab ───────────────────────────────────────
    run_one_test(b, slot, "map15_06b_laje_zigzag",
      () -> begin
        pts_serra(p0, dentes, comprimento, altura) =
          dentes == 0 ?
            [p0] :
            let p1 = p0 + vxy(comprimento/2, altura)
                p2 = p0 + vx(comprimento)
              [p0, p1, pts_serra(p2, dentes - 1, comprimento, altura)...]
            end
        laje_zigzag(p, l, c, e, a, n) =
          let dc = c/n,
              pts_s = pts_serra(p, n, dc, a),
              pts = [pts_s..., pts_s[end] + vy(l), pts_s[begin] + vy(l)]
            extrusion(surface_polygon(pts), e)
          end
        laje_zigzag(u0(), 5.0, 10.0, 0.3, 0.5, 8)
      end,
      nothing,
      verify)

    # ── map15-07a: helical spring (mola) ─────────────────────────────
    run_one_test(b, slot, "map15_07a_mola",
      () -> begin
        helice(p, r, h, v, n) =
          map_division(ti -> p + vcyl(r, ti, h*ti/2*pi),
                       0, v*2*pi, v*n)
        mola(p, r0, r1, h, v, n) =
          sweep(spline(helice(p, r0, h, v, n)),
                surface_circle(u0(), r1))
        mola(u0(), 3.0, 0.3, 0.5, 4, 30)
      end,
      nothing,
      verify)

    # ── map15-07b: arquimedes-spiral wall ────────────────────────────
    run_one_test(b, slot, "map15_07b_parede_espiral_arquimedes",
      () -> begin
        espiral_arquimedes(p, alfa, t0, t1, n) =
          map(t -> p + vpol(alfa*t, t), division(t0, t1, n))
        parede_curva(espessura, altura, curva) =
          let seccao = surface_rectangle(xy(0, 0), xy(espessura, altura))
            sweep(curva, seccao)
          end
        parede_espiral_arquimedes(espessura, altura, p, alfa, t0, t1, n) =
          parede_curva(espessura, altura,
                       spline(espiral_arquimedes(p, alfa, t0, t1, n)))
        parede_espiral_arquimedes(0.5, 5.0, xy(0, 0), 1.0, 0, 8*pi, 80)
      end,
      nothing,
      verify)

    # ── map15-08b: explicit surface_grid expression ──────────────────
    run_one_test(b, slot, "map15_08b_surface_grid_expr",
      () -> surface_grid(
              map_division((x_, y_) -> xyz(x_, y_, 1.5*exp(-(x_-(3+0.5*sin(y_)))^2)),
                           0, 2*pi, 20,
                           0, 2*pi, 20), false, false, true, true),
      nothing,
      verify)

    # ── map30-01a: triangle with 3 corner circles ────────────────────
    run_one_test(b, slot, "map30_01a_triangulo_circulos",
      () -> begin
        triangulo_circulos(p, b_, a, r) =
          let p1 = p + vx(b_),
              p2 = p + vxy(b_/2, a)
            polygon(p, p1, p2)
            circle(p, r)
            circle(p1, r)
            circle(p2, r)
          end
        triangulo_circulos(u0(), 5.0, 4.0, 0.5)
      end,
      nothing,
      verify)

    # ── map30-01b: triangle with midpoints triangle ──────────────────
    run_one_test(b, slot, "map30_01b_triangulo_intermedio",
      () -> begin
        intermediate_point(p, q) =
          xyz((p.x + q.x)/2, (p.y + q.y)/2, (p.z + q.z)/2)
        triangulo_intermedio(p1, p2, p3) =
          let q1 = intermediate_point(p1, p2)
              q2 = intermediate_point(p2, p3)
              q3 = intermediate_point(p3, p1)
            polygon(p1, p2, p3)
            polygon(q1, q2, q3)
          end
        triangulo_intermedio(u0(), xy(5, 0), xy(2.5, 4))
      end,
      nothing,
      verify)

    # ── map30-02a: shrinking-rectangles area sum ─────────────────────
    run_one_test(b, slot, "map30_02a_soma_areas",
      () -> begin
        soma_areas(p, c, l, f, n) =
          if n == 0
            nothing
          else
            rectangle(p, c, l)
            soma_areas(p + vx(c), c*f, l*f, f, n - 1)
          end
        soma_areas(u0(), 4.0, 4.0, 0.7, 8)
      end,
      nothing,
      verify)

    # estrela (star) — uses line, no random
    run_one_test(b, slot, "map30_02a_estrela",
      () -> begin
        estrela(p, ri, re, fi, dfi) =
          if fi >= 2pi
            nothing
          else
            line(p + vpol(ri, fi),
                 p + vpol(re, fi + dfi/2),
                 p + vpol(ri, fi + dfi))
            estrela(p, ri, re, fi + dfi, dfi)
          end
        estrela(u0(), 2.0, 4.0, 0.0, 2pi/8)
      end,
      nothing,
      verify)

    # ── map30-02b: rotated squares + nested circles + cylinder spiral ─

    run_one_test(b, slot, "map30_02b_quadrados_rodados",
      () -> begin
        quadrados_rodados(p, d, n, a, da) =
          if n == 0
            nothing
          else
            regular_polygon(4, p, d/2, a)
            quadrados_rodados(p + vx(d), d, n - 1, a + da, da)
          end
        quadrados_rodados(u0(), 2.0, 8, 0.0, π/12)
      end,
      nothing,
      verify)

    # NOTE: map30-02b's `circulo_no_circulo` solution intentionally uses
    # strict equality `if r == 1` and the question asks students to
    # explain why this never terminates. Excluded from the test set.

    run_one_test(b, slot, "map30_02b_espiral_cilindros",
      () -> begin
        espiral_cilindros(p, r, h, a, da, fr, fh, n) =
          if n == 0
            nothing
          else
            cylinder(p, r, h)
            espiral_cilindros(p + vpol(r + r*fr, a),
                              r*fr, h*fh, a + da, da, fr, fh, n - 1)
          end
        espiral_cilindros(u0(), 5.0, 10.0, 0.0, π/4, 1.2, 1.2, 12)
      end,
      nothing,
      verify)

    # ── map30-03a: flower (CSG of spheres) ───────────────────────────
    run_one_test(b, slot, "map30_03a_flor",
      () -> begin
        flor(p, r, n) =
          let petalas = [sphere(p + vpol(r, a), r) for a in 0:2pi/n:2pi]
            subtraction(union(petalas), sphere(p, r))
          end
        flor(u0(), 1.0, 5)
      end,
      nothing,
      verify)

    # ── map30-03b: orange wedges (slice CSG) ─────────────────────────
    run_one_test(b, slot, "map30_03b_laranja",
      () -> begin
        gomo(p, r, ϕ, Δϕ) =
          slice(slice(sphere(p, r),
                      p, vpol(1, ϕ - π/2)),
                p, vpol(1, ϕ + Δϕ + π/2))
        laranja(p, r, s, n) =
          let Δϕ = 2π/n - s
            for ϕ in division(0, 2π, n)
              gomo(p, r, ϕ, Δϕ)
            end
          end
        laranja(u0(), 3.0, 0.05, 8)
      end,
      nothing,
      verify)

    # ── map30-04a: revolved vase profile ─────────────────────────────
    run_one_test(b, slot, "map30_04a_vaso",
      () -> begin
        perfil_vaso(p, r0, r1, h, e) =
          let p1 = p + vx(r0), p2 = p + vxz(r1, h),
              p3 = p2 - vx(e), p4 = p1 + vxz(-e, e),
              p5 = p + vz(e)
            surface(line(p, p1, p2, p3, p4, p5, p))
          end
        vaso(p, r0, r1, h, e) = revolve(perfil_vaso(p, r0, r1, h, e), p)
        vaso(u0(), 0.7, 1.2, 1.0, 0.1)
      end,
      nothing,
      verify)

    # ── map30-04b: pillar of radial circles ──────────────────────────
    run_one_test(b, slot, "map30_04b_pilar_circulos_radiais",
      () -> begin
        circulos_radiais(p, r0, r1, n) =
          union([surface_circle(p + vpol(r0, i), r1)
                 for i in division(0, 2pi, n, false)])
        pilar_circulos_radiais(p, r0, r1, n, h) =
          extrusion(circulos_radiais(p, r0, r1, n), h)
        pilar_circulos_radiais(u0(), 1.0, 0.5, 6, 7.0)
      end,
      nothing,
      verify)

    # ── map30-05a: hyperboloid tower (rotated triangles synth call) ──
    run_one_test(b, slot, "map30_05a_torre_hiperboloide",
      () -> begin
        pontos_circulares(p, r, fi, n) =
          [p + vpol(r, fi + dfi) for dfi in division(0, 2pi, n, false)]
        torre_hiperboloide(p, r, rb, rt, h, fi, n) =
          let pts1 = pontos_circulares(p, rb, 0, n),
              pts2 = pontos_circulares(p + vz(h), rt, fi, n)
            [cylinder(p_, 0.1, q) for (p_, q) in zip(pts1, pts2)]
          end
        torre_hiperboloide(u0(), 0.1, 4.0, 4.0, 8.0, π/4, 12)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "map30_05a_triangulos_rodados",
      () -> begin
        triangulos_rodados(p, r, a, n) =
          let da = a/n
            map_division((i, j) -> regular_polygon(3,
                                                   p + vxy(2*r*i, 2*r*j),
                                                   r,
                                                   π/4 + da*(i + j),
                                                   true),
                         0, n, n,
                         0, n, n)
          end
        triangulos_rodados(u0(), 0.4, π/4, 6)
      end,
      nothing,
      verify)

    # ── map30-05b: möbius strip with spheres ─────────────────────────
    run_one_test(b, slot, "map30_05b_moebius_esferas",
      () -> begin
        posicao_moebius(p, r, l, u, v) =
          p + vcyl(r + l*v*cos(u/2),
                   u,
                   l*v*sin(u/2))
        posicoes_moebius(p, r, l, m, n) =
          map_division((u, v) -> posicao_moebius(p, r, l, u, v),
                       0, 4pi, m,
                       0, 1, n)
        for ps in posicoes_moebius(u0(), 2.0, 1.0, 60, 4)
          for p in ps
            sphere(p, 0.05)
          end
        end
      end,
      nothing,
      verify)

    # ── Exam-derived exercises ──────────────────────────────────────
    # Solutions extracted from .tex exam files in
    # ~/Vault/AML/Teaching/ElementosPedagogicos/Arquitectura/exames/.
    # Each test reproduces the JuliaCode solution verbatim and is
    # architecturally distinct from the aula/map tests above.

    run_one_test(b, slot, "exame_2024_escadas",
      () -> begin
        escadas_blocos(p, c, l, e, h, n) =
          if n == 0
            nothing
          else
            box(p, p + vxyz(c, l, h))
            escadas_blocos(p + vx(c), c, l, e, h + e, n - 1)
          end
        escadas(p, c, l, e, n) = escadas_blocos(p, c, l, e, e, n)
        escadas(u0(), 0.25, 2.0, 0.16, 10)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "exame_2024_promenade",
      () -> begin
        coluna_portico(p, r, a) = cylinder(p, r, a)
        lintel(p, c, l, a) = box(p + vxyz(c/-2.0, l/-2.0, a/-2.0), c, l, a)
        portico(p, rc, ac, dc, al) = begin
          coluna_portico(p, rc, ac)
          coluna_portico(p + vxyz(dc, 0, 0), rc, ac)
          lintel(p + vxyz(dc/2.0, 0, ac + al/2.0), dc + rc + rc, rc + rc, al)
        end
        promenade(p, rc, ac, dc, al, dp, n) =
          if n == 0
            nothing
          else
            portico(p, rc, ac, dc, al)
            promenade(p + vy(dp), rc, ac, dc, al, dp, n - 1)
          end
        promenade(u0(), 0.5, 4.0, 5.0, 0.4, 2.0, 4)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "exame_2024_emaranhado",
      () -> begin
        emaranhado(p, d, n) =
          if n == 0
            nothing
          else
            let p1 = p + vpol(random_range(0, d), random(8)*pi/4)
              line(p, p1)
              emaranhado(p1, d, n - 1)
            end
          end
        set_random_seed(12345)
        emaranhado(u0(), 1.0, 80)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "exame_2025_lajes_serra",
      () -> begin
        pontos_serra(p, c, a, n) =
          n == 0 ? [p] :
            [p, p + vxy(c/2, a),
             pontos_serra(p + vx(c), c, a, n - 1)...]
        laje_serra(p, c, l, e, a, n) =
          let d = c/n
            extrusion(
              surface_polygon(
                [pontos_serra(p, d, a, n)...,
                 p + vxy(c, l),
                 p + vy(l)]),
              vz(e))
          end
        lajes_serra(p, c, l, e, a, n, h) =
          if n == 0
            nothing
          else
            laje_serra(p, c, l, e, a, n)
            lajes_serra(p + vz(h), c, l, e, a, n - 1, h)
          end
        lajes_serra(u0(), 6.0, 4.0, 0.2, 0.5, 5, 1.0)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "exame_2025_cobertura_bosjes",
      () -> begin
        pontos_bosjes(p, c, l, h, n, m) =
          map_division((u, v) ->
              p + vxyz(u, v,
                       h/2*(1 + cos(u/c*4pi)*cos(v/l*2pi))),
              0, c, n,
              0, l, m)
        cobertura_bosjes(p, c, l, h, n, m, e) =
          thicken(
            surface_grid(pontos_bosjes(p, c, l, h, n, m)),
            e)
        cobertura_bosjes(u0(), 12.0, 12.0, 4.0, 24, 24, 0.2)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "exame_2025_walkie_talkie",
      () -> begin
        parabola_yz(p, α, t) = p + vyz(α*t^2, t)
        pontos_parabola_yz(p, α, z0, z1, n) =
          map_division(z -> parabola_yz(p, α, z), z0, z1, n)
        coef_parabola(h, lb, lt) = (lt - lb)/h^2
        walkie_talkie(p, lb, lt, h, n) =
          let pts = pontos_parabola_yz(p - vx(lb/2),
                                       coef_parabola(h, lb, lt), 0, h, n)
            # Original exam solution closed the back-line at `p`, jumping
            # x from -lb/2 (parabola plane) to 0 → non-planar boundary,
            # rejected by AutoCAD's `Region.CreateFromCurves` and Rhino's
            # `SurfaceFrom`. Closing at `p - vx(lb/2)` keeps every vertex
            # on the parabola plane.
            extrusion(surface(spline(pts),
                              line(pts[end], p + vxyz(-lb/2, -lb, h),
                                   p + vxy(-lb/2, -lb), p - vx(lb/2))),
                      vx(lb))
          end
        walkie_talkie(u0(), 4.0, 8.0, 16.0, 30)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "exame_2025_pilares",
      () -> begin
        pontos_bosjes(p, c, l, h, n, m) =
          map_division((u, v) ->
              p + vxyz(u, v, h/2*(1 + cos(u/c*4pi)*cos(v/l*2pi))),
              0, c, n, 0, l, m)
        # The exam solution emits a cylinder for every grid point, but the
        # Bosjes formula evaluates to z=0 at the corners (cos*cos = -1) and
        # `cylinder(xy(pt.x, pt.y), r, pt)` then has zero height, raising
        # `vxyz(0,0,0) too small to unitize`. Skip degenerate points.
        pilares(ptss, r) =
          for pts in ptss
            for pt in pts
              if pt.z > 1e-6
                cylinder(xy(pt.x, pt.y), r, pt)
              end
            end
          end
        pilares(pontos_bosjes(u0(), 12.0, 12.0, 4.0, 6, 6), 0.1)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "ep_especial_2024_espiral_recta",
      () -> begin
        espiral_recta(p, l, alfa, d_l) =
          if l <= d_l
            nothing
          else
            let p1 = p + vpol(l, alfa)
              line(p, p1)
              espiral_recta(p1, l - d_l, alfa + pi/2, d_l)
            end
          end
        espiral_recta(u0(), 10.0, pi/2, 0.1)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "ep_especial_2024_curva_atractor",
      () -> begin
        move_posicao(p, a) =
          p + (p - a)/2/distance(p, a)*5^(-distance(p, a))
        curva_atractor(p, q, a, n) =
          spline(map_division(t -> move_posicao(p + (q - p)*t, a),
                              0, 1, n))
        curva_atractor(xy(0, 0), xy(10, 0), xy(5, 3), 40)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "recurso_2022_abertura",
      () -> begin
        posicao_intermedia(p, m) = p + (m - p)/2
        centro_quadrangulo(p1, p2, p3, p4) =
          posicao_intermedia(
            posicao_intermedia(p1, p3),
            posicao_intermedia(p2, p4))
        abertura(p1, p2, p3, p4) =
          let m = centro_quadrangulo(p1, p2, p3, p4),
              p1m = posicao_intermedia(p1, m),
              p2m = posicao_intermedia(p2, m),
              p3m = posicao_intermedia(p3, m),
              p4m = posicao_intermedia(p4, m)
            surface_polygon(p1, p2, p2m, p1m)
            surface_polygon(p2, p3, p3m, p2m)
            surface_polygon(p3, p4, p4m, p3m)
            surface_polygon(p4, p1, p1m, p4m)
          end
        abertura(xyz(0, 0, 0), xyz(4, 0, 0), xyz(4, 4, 0), xyz(0, 4, 0))
      end,
      nothing,
      verify)

    run_one_test(b, slot, "recurso_2023_piao",
      () -> begin
        perfil_piao(p, r, h) =
          let p0 = p + vz(h - r),
              p1 = p - vx(r),
              p2 = p - vz(r)
            spline(p0, p1, p2)
          end
        piao(p, r, h) =
          revolve(perfil_piao(p, r, h), p, vz(1), 0, 2pi)
        piao(u0(), 1.0, 3.0)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "recurso_2023_toro_esferas",
      () -> begin
        pontos_toro(p, r0, r1, m, n) =
          map_division((u, v) -> p + vxyz(cos(v)*(r0 + r1*cos(u)),
                                           sin(v)*(r0 + r1*cos(u)),
                                           r1*sin(u)),
                       0, 2pi, m,
                       2pi, 0, n)
        itera_quadrangulos(f, ptss) =
          [[f(p0, p1, p2, p3)
            for (p0, p1, p2, p3)
            in zip(pts0[1:end-1], pts1[1:end-1], pts1[2:end], pts0[2:end])]
           for (pts0, pts1) in zip(ptss[1:end-1], ptss[2:end])]
        toro_esferas(p, r0, r1, re, m, n) =
          itera_quadrangulos((p1, p2, p3, p4) -> sphere(quad_center(p1, p2, p3, p4), re),
                             pontos_toro(p, r0, r1, m, n))
        toro_esferas(u0(), 8.0, 2.0, 0.4, 24, 12)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "recurso_2024_quadrado_arcos",
      () -> begin
        arco(p, r, s) =
          r <= s ?
            arc(p, r, 0, pi/2) :
            let a = acos(s/r)
              arc(p, r, a, pi/2 - 2a)
            end
        quadrado_arcos(p, s, n) =
          let dr = sqrt(2)*s/n
            for i in 1:n
              arco(p, dr*i, s)
            end
          end
        quadrado_arcos(u0(), 10.0, 30)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "recurso_2024_esfera_esburacada",
      () -> begin
        esfera_esburacada(p, r0, r1_min, r1_max, n) =
          subtraction(sphere(p, r0),
                      union([sphere(p + vsph(r0, random_range(0, 2*pi),
                                                  random_range(0, pi)),
                                    random_range(r1_min, r1_max))
                             for i in 1:n]))
        set_random_seed(12345)
        esfera_esburacada(u0(), 5.0, 0.4, 1.2, 8)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "recurso_2024_trajectoria_orbital",
      () -> begin
        posicao_orbital(p, r0, r1, omega0, omega1, t) =
          p + vpol(r0, omega0*t) + vpol(r1, omega1*t)
        trajectoria_orbital(p, r0, r1, omega0, omega1, ti, tf, n) =
          spline(map_division(t -> posicao_orbital(p, r0, r1, omega0, omega1, t),
                              ti, tf, n))
        trajectoria_orbital(u0(), 5.0, 1.0, 1.0, 7.0, 0, 4*pi, 200)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "recurso_2024_vaso_grid",
      () -> begin
        pontos_vaso(p, r, h, a, omega, n, m) =
          surface_grid(map_division((z, fi) -> p + vcyl(r + a*sin(omega*z), fi, z),
                                    0, h, n,
                                    0, 2*pi, m))
        pontos_vaso(u0(), 2.0, 6.0, 0.3, 4.0, 30, 30)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "recurso_2025_circulos_no_circulo",
      () -> begin
        circulos_no_circulo(p, ri, re, rl, n) =
          if n == 0
            nothing
          else
            let r = random_range(ri, re)
              circle(p + vpol(r, random_range(0, 2*pi)), rl - r)
              circulos_no_circulo(p, ri, re, rl, n - 1)
            end
          end
        set_random_seed(12345)
        circulos_no_circulo(u0(), 0.5, 4.0, 5.0, 20)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "recurso_2025_sol",
      () -> begin
        raios(p, r1, r2, phi, dphi, n) =
          if n == 0
            nothing
          else
            line(p + vpol(r1, phi), p + vpol(r2, phi))
            raios(p, r1, r2, phi + dphi, dphi, n - 1)
          end
        sol(p, r1, r2, n) =
          begin
            circle(p, r1)
            raios(p, r1, r2, 0, 2pi/n, n)
          end
        sol(u0(), 2.0, 3.5, 24)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "recurso_2025_chess_board",
      () -> begin
        chess_row(p, l, h, n) =
          if n == 0
            nothing
          else
            box(p, l, l, h)
            chess_row(p + vx(l), l, -h, n - 1)
          end
        chess_board(p, l, h, n, m) =
          if n == 0
            nothing
          else
            chess_row(p, l, h, m)
            chess_board(p + vy(l), l, -h, n - 1, m)
          end
        chess_board(u0(), 1.0, 0.3, 8, 8)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "recurso_2025_esfera_sinusoidal",
      () -> begin
        esfera_sinusoidal(p, re, ri, a, omega, n, m) =
          map_division(
            psi -> sweep(
              closed_spline(
                map_division(t -> p + vsph(re, t, psi + a*sin(psi)*sin(omega*t)),
                             0, 2*pi, n, false)),
              surface_circle(u0(), ri)),
            0.7*re*ri*a, pi - 0.7*re*ri*a, m)
        # ri reduced from 0.15 → 0.02, n raised from 24 → 120, AND wave
        # amplitude `a` reduced from 0.15 → 0.05. The closed-spline path's
        # curvature comes from both ri and the latitude-perturbation
        # amplitude; AutoCAD's `Solid3d.CreateSweptSolid` returns
        # `eGeneralModelingFailure` when either is too aggressive relative
        # to the section's `ri`.
        esfera_sinusoidal(u0(), 5.0, 0.02, 0.05, 6, 120, 6)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "first_test_2019_slanted_lines",
      () -> begin
        slanted_lines(p, d, n) =
          let s = d*2/n,
              q = p + vxy(d, d)
            for i in 1:div(n, 2)
              line(p + vx(i*s), p + vy(i*s))
            end
            for i in 1:div(n, 2) - 1
              line(q - vx(i*s), q - vy(i*s))
            end
          end
        rectangle(u0(), 10, 10)
        slanted_lines(u0(), 10, 10)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "first_test_2019_random_lines",
      () -> begin
        random_lines(n, l, w, h) =
          if n == 0
            nothing
          else
            let p = xy(random_range(l, w - l), random_range(l, h - l))
              line(p, p + vpol(l, random_range(0, 2pi)))
              random_lines(n - 1, l, w, h)
            end
          end
        set_random_seed(12345)
        rectangle(u0(), 5, 3)
        random_lines(200, 0.25, 5.0, 3.0)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "first_test_2019_connect_random_points",
      () -> begin
        random_points(n, w, h) =
          if n == 0
            Loc[]
          else
            [xy(random_range(0, w), random_range(0, h)),
             random_points(n - 1, w, h)...]
          end
        connect_points(pts) =
          if pts == Loc[]
            nothing
          else
            let p = pts[1]
              for q in pts[2:end]
                line(p, q)
              end
              connect_points(pts[2:end])
            end
          end
        set_random_seed(12344)
        connect_points(random_points(15, 10.0, 5.0))
      end,
      nothing,
      verify)

    run_one_test(b, slot, "repescagem_2019_lines",
      () -> begin
        lines(p, rho, alpha, n) =
          n == 0 ? nothing :
            let p1 = p + vpol(rho*random_range(0.1, 1.0), alpha),
                pm = p + (p1 - p)*random(1.0)
              line(p, p1)
              lines(pm, rho, alpha + pi/2, n - 1)
            end
        set_random_seed(12345)
        lines(u0(), 10, 0.2, 100)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "repescagem_2019_faixas",
      () -> begin
        faixas(p, q, d, n) =
          n == 0 ? nothing :
            let p1 = p + vx(d),
                q1 = q + vy(d)
              surface_polygon(p, q, q1, p1)
              faixas(p1 + vx(d), q1 + vy(d), d, n - 1)
            end
        faixas(u0(), xy(0, 1), 1.0, 6)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "repescagem_2019_cubo_derivado",
      () -> begin
        cubo_derivado(p, l) =
          let v = vxyz(l, l, l)/2
            subtraction(box(p - v, p + v),
                        box(p, p + v))
          end
        cubo_derivado(u0(), 5)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "teste_2020_coronavirus",
      () -> begin
        pontos_espinho(p, a, rs, ss) =
          [p + vpol(rs[1], a) + vpol(ss[1], a - pi/2),
           p + vpol(rs[2], a) + vpol(ss[2], a - pi/2),
           p + vpol(rs[3], a) + vpol(ss[3], a - pi/2),
           p + vpol(rs[4], a),
           p + vpol(rs[3], a) + vpol(ss[3], a + pi/2),
           p + vpol(rs[2], a) + vpol(ss[2], a + pi/2),
           p + vpol(rs[1], a) + vpol(ss[1], a + pi/2)]
        pontos_espinhos(p, a, da, n, rs, ss) =
          n == 0 ? Loc[] :
            [pontos_espinho(p, a, rs, ss)...,
             pontos_espinhos(p, a + da, da, n - 1, rs, ss)...]
        coronavirus(p, n, rs, ss) =
          closed_spline(
            pontos_espinhos(p, 0, 2pi/n, n, rs, ss))
        coronavirus(u0(), 8, [5.0, 7.0, 8.0, 9.0], [1.0, 0.5, 1.0])
      end,
      nothing,
      verify)

    run_one_test(b, slot, "teste_2026_espiral_teodoro",
      () -> begin
        sqrt_spiral(p, α, i, j) =
          i >= j ? nothing :
            let p1 = p + vpol(sqrt(i), α),
                p2 = p1 + vpol(1, α + π/2)
              polygon(p, p1, p2)
              sqrt_spiral(p, pol_phi(p2 - p), i + 1, j)
            end
        espiral_teodoro(p, α, n) = sqrt_spiral(p, α, 1, n)
        espiral_teodoro(u0(), 0.0, 15)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "teste_2026_montanhas",
      () -> begin
        montanhas(p, c, a, n) =
          if n == 0
            nothing
          else
            let c1 = c*random_range(0.3, 1),
                a1 = a*random_range(0.3, 1),
                p2 = p + vx(c1)
              spline(p, p + vxy(c1/2, a1), p2)
              montanhas(p2, c, a, n - 1)
            end
          end
        set_random_seed(12345)
        montanhas(u0(), 1.0, 1.5, 12)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "second_test_2019_cobertura_pavilhao",
      () -> begin
        parametrica_abobada(p, lx, ly, lz) =
          (u, v) -> p + vxyz(lx*u, ly*v, lz*sin(u*pi/2)*sin(v*pi))
        abobada_pavilhao(p, lx, ly, lz, e, n, m) =
          thicken(
            surface_grid(
              map_division(parametrica_abobada(p, lx, ly, lz),
                           0, 1, n, 0, 1, m)),
            e)
        cobertura_pavilhao(p, lx, ly, lz, e, n, m, nabobadas) =
          for i in 0:nabobadas - 1
            abobada_pavilhao(p + vx(lx*i), lx, ly, lz, e, n, m)
          end
        # lz reduced from 2.0 → 0.1 to keep the double-sinusoidal vault
        # gentle enough for AutoCAD's `Surface.Thicken`, which fails with
        # `eGeneralModelingFailure` when the surface curvature is high
        # relative to the thickness.
        cobertura_pavilhao(u0(), 4.0, 8.0, 0.1, 0.1, 16, 24, 3)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "second_test_2019_barras_superficie",
      () -> begin
        sinusoide(a, omega, fi, x) = a*sin(omega*x + fi)
        pontos_parede(p, a, omega, phi, c, h, n, m) =
          map_division(
            (x, z) -> p + vxyz(x, sinusoide(a*z/h, omega, phi, x), z),
            0, c, n,
            0, h, m)
        transposta(ptts) =
          [map(pts -> pts[i], ptts)
           for i in 1:length(ptts[1])]
        barra_curva(pts, dx, dy) =
          sweep(spline(pts),
                surface_rectangle(xy(-dx/2, -dy/2), dx, dy))
        barras_superficie(ptts, dx, dy) =
          begin
            map(pts -> barra_curva(pts, dx, dy), ptts)
            map(pts -> barra_curva(pts, dx, dy), transposta(ptts))
          end
        barras_superficie(pontos_parede(u0(), 0.5, 1.0, 0.0, 6.0, 4.0, 8, 6),
                          0.1, 0.1)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "repescagem_2019_coluna_gaudi",
      () -> begin
        prisma_torcido(p, r, n, h, dfi, e) =
          sweep(line(p, p + vz(h)),
                surface_regular_polygon(n, u0(), r),
                dfi,
                e)
        coluna_gaudi(p, r, n, h, dfi, e, combina) =
          combina(
            prisma_torcido(p, r, n, h,  dfi, e),
            prisma_torcido(p, r, n, h, -dfi, e))
        coluna_gaudi(u0(), 1.0, 4, 10.0, pi/8, 0.9, union)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "repescagem_2020_vigas_moebius",
      () -> begin
        posicoes_moebius(p, r, l, m, n) =
          map_division(
            (u, v) -> p + vcyl(r + l*v*cos(u/2), u, l*v*sin(u/2)),
            0, 4pi, m,
            0, 1, n)
        viga(pts, c, a) =
          sweep(spline(pts),
                surface_rectangle(xy(-c/2, -a/2), c, a))
        vigas_moebius(p, r, l, m, n, c, a) =
          map(pts -> viga(pts, c, a),
              posicoes_moebius(p, r, l, m, n))
        # Square section (c == a) so the per-frame transform from the
        # Möbius path is a uniform scaling, avoiding AutoCAD's
        # `eCannotScaleNonUniformly` rejection in `Entity.TransformBy`.
        vigas_moebius(u0(), 2.0, 1.0, 60, 4, 0.05, 0.05)
      end,
      nothing,
      verify)

    run_one_test(b, slot, "repescagem_2020_aberturas_moebius",
      () -> begin
        posicoes_moebius(p, r, l, m, n) =
          map_division((u, v) -> p + vcyl(r + l*v*cos(u/2), u, l*v*sin(u/2)),
                       0, 4pi, m,
                       0, 1, n)
        itera_quadrangulos(f, ptss) =
          [[f(p0, p1, p2, p3)
            for (p0, p1, p2, p3)
            in zip(pts0[1:end-1], pts1[1:end-1], pts1[2:end], pts0[2:end])]
           for (pts0, pts1) in zip(ptss[1:end-1], ptss[2:end])]
        posicao_intermedia(p, m) = p + (m - p)/2
        centro_quadrangulo(p1, p2, p3, p4) =
          posicao_intermedia(posicao_intermedia(p1, p3),
                             posicao_intermedia(p2, p4))
        abertura(p1, p2, p3, p4) =
          let m = centro_quadrangulo(p1, p2, p3, p4),
              p1m = posicao_intermedia(p1, m),
              p2m = posicao_intermedia(p2, m),
              p3m = posicao_intermedia(p3, m),
              p4m = posicao_intermedia(p4, m)
            surface_polygon(p1, p2, p2m, p1m)
            surface_polygon(p2, p3, p3m, p2m)
            surface_polygon(p3, p4, p4m, p3m)
            surface_polygon(p4, p1, p1m, p4m)
          end
        aberturas_moebius(p, r, l, m, n) =
          itera_quadrangulos(abertura, posicoes_moebius(p, r, l, m, n))
        aberturas_moebius(u0(), 2.0, 0.6, 40, 4)
      end,
      nothing,
      verify)
  end
