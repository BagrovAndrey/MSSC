Конечно. Ниже — handoff, который можно просто вставить в новый чат.

---

# MSSC / structural complexity — handoff summary

## 1. Цель проекта сейчас

Мы сознательно хотим **закончить exploratory metric design** и прийти к статье-MVP:

1. зафиксировать определение structural complexity для одного конкретного image observer;
2. показать controlled sanity checks на synthetic patterns;
3. дать 1–2 физических приложения, наиболее очевидно — 2D Ising transition, возможно затем DM/labyrinth/skyrmion patterns.

Пока **не** пытаемся делать универсальную MSSC для графов/time series и т.п.

Observer фиксирован:

* non-overlapping dyadic `2x2` block averaging;
* nearest-neighbor lifting;
* Haar-like detail channels;
* grayscale как основной режим.

---

# 2. Базовый RG / Haar observer

Для RG trajectory

$$
f_0\to f_1\to f_2\to\ldots
$$

старый MSSC detail residual:

$$
d_k=f_k-Uf_{k+1},
$$

$$
C_k=\frac12\langle d_k^2\rangle,
$$

$$
C_{\rm detail}=\sum_k C_k.
$$

Для scalar `2x2` блока

$$
\begin{pmatrix}
a&b\\
c&d
\end{pmatrix}
$$

Haar channels:

$$
h_x=(a+c-b-d)/4,
$$

$$
h_y=(a+b-c-d)/4,
$$

$$
h_{xy}=(a-b-c+d)/4.
$$

Локальная channel energy:

$$
E_{k,B,\alpha}=\frac12h_{k,B,\alpha}^2.
$$

Эти energies lift'ятся обратно на original grid:

$$
E_{k,\alpha}(x).
$$

Channel label удобно считать

$$
c=(k,\alpha).
$$

---

# 3. Information-theory decomposition

Это сейчас одна из наиболее чистых частей конструкции.

Пусть задан некоторый organization gate и получены локальные organized weights:

$$
W_c(x)=W_{k,\alpha}(x).
$$

Определяем

$$
W(x)=\sum_cW_c(x),
$$

$$
Z=\sum_{x,c}W_c(x),
$$

$$
\bar W=\frac{Z}{N_{\rm pix}}.
$$

После нормировки

$$
p(x,c)=\frac{W_c(x)}{Z}.
$$

Тогда:

$$
p(c|x)=\frac{W_c(x)}{W(x)},
$$

$$
p(c)=\sum_xp(x,c).
$$

### Nested branch

$$
H_{\rm nested}=H(C|X).
$$

Это richness scale-orientation RG-history **внутри одной локальной области**.

Extensive version:

$$
J_{\rm nested}
=
\bar W H_{\rm nested}.
$$

### Heterogeneous branch

$$
I_{\rm hetero}=I(X;C)
=
H(C)-H(C|X).
$$

Это diversity **между local RG histories в разных местах изображения**.

Extensive version:

$$
J_{\rm hetero}
=
\bar W I_{\rm hetero}.
$$

### Total

$$
H_{\rm struct}=H(C)
=
H_{\rm nested}+I_{\rm hetero},
$$

и

$$
\boxed{
J_{\rm struct}
=
J_{\rm nested}+J_{\rm hetero}
=
\bar W H_{\rm struct}.
}
$$

Это точное information-theoretic decomposition, а не эвристическая сумма разных метрик.

Интерпретация:

* `Jnested`: rich local multiscale histories;
* `Jhetero`: different structural histories in different parts of the image;
* `Jstruct`: оба вида structural diversity.

Patchwork перестали считать failure mode: это legitimate complexity, почти целиком heterogeneous.

---

# 4. Старый organization gate и почему от него отказались

Изначально использовалась локальная orientation coherence:

$$
q^{\rm orient}_k(x),
$$

и

$$
W_{k,\alpha}(x)
=
q^{\rm orient}_k(x)E_{k,\alpha}(x).
$$

Этот старый вариант назывался `JlocQ`; в новой терминологии это был `Jnested` с orientation gate.

Проблема: orientation gate считает

> organization ≈ local orientational smoothness.

Он отлично любит smooth wavy stripes и сильно штрафует:

* corners;
* branching;
* rapidly changing directions;
* fractal boundaries.

### Ключевой q-ablation

Сравнивали:

$$
W=E,
\qquad
W=\sqrt q\,E,
\qquad
W=qE.
$$

Получили:

```text
                  Jstruct
              q0      qsqrt      q1
patchwork    0.693     0.688     0.685
fractal      1.506     0.375     0.214
wavy         0.997     0.663     0.561
noise        0.926     0.175     0.091
```

При `q=0`:

$$
fractal>wavy,
$$

но noise слишком высокий.

При старом `q`:

$$
wavy\gg fractal.
$$

Причина была очень ясной:

```text
Wbar fractal = 0.085
Wbar wavy    = 0.333
```

и retained detail fraction:

```text
fractal ≈ 17%
wavy    ≈ 67%
```

Причем без q:

$$
H_{\rm nested}^{fractal}=2.136,
$$

$$
H_{\rm nested}^{wavy}=1.482.
$$

То есть сама entropy правильно видела fractal как более богатый. Именно orientation gate портил ranking.

---

# 5. Проверка гипотезы про multiscale replication / lifting

Была гипотеза, что wavy stripes получают большую complexity, потому что одна smooth boundary при coarse-graining начинает относиться к огромному числу descendant pixels.

Проверили число active RG scales для каждой original-space точки, используя raw lifted energy \(E\), без q.

Результат:

```text
                 mean n_E   median   p90
fractal             6.97       7      9
wavy                5.11       5      7
```

То есть **fractal имеет более длинные active RG histories**, не wavy.

Поэтому lifting сам по себе не создавал неправильный ranking.

Но у wavy есть характерный эффект:

на scales \(k=4,5\)

```text
effective support ≈ 0.993, 0.999
```

то есть detail energy основного stripe motif практически покрывает всю картинку.

Вывод:

* multiscale support wavy действительно большой на нескольких preferred scales;
* но это скорее **усилитель** проблемы;
* критическим culprit был orientation gate.

---

# 6. Новый organization gate: energy coherence

Появилась новая идея: gate должен отличать

> structured detail vs noise,

но **не** требовать alignment Haar directions.

На native RG block grid, ДО lifting:

$$
e_{k,B}
=
\sum_\alpha h_{k,B,\alpha}^2.
$$

Для horizontal + vertical nearest-neighbor block pairs считаем **centered Pearson correlation**:

$$
\rho^E_k
=
{\rm corr}(e_B,e_{B'}).
$$

Важно: Pearson вычитает mean. Поэтому у white noise может быть высокая средняя detail energy, но если fluctuations независимы:

$$
\rho^E_k\approx0.
$$

Gate:

$$
Q^E_k=\max(\rho^E_k,0).
$$

Canonical candidate weights:

$$
\boxed{
W_{k,\alpha}(x)
=
Q^E_k E_{k,\alpha}(x)
}
$$

где \(Q^E_k\) — **один scalar на scale**, а \(E\) уже lifted locally.

Очень важно:

* `Qenergy` считается **до lifting**;
* иначе lifting сам создает artificial correlations;
* если variance native block energies почти zero, Pearson undefined → `Qenergy=NaN/undefined`, не 1;
* на очень coarse scales мало neighbor pairs, поэтому введен safeguard вроде `min_qenergy_pairs = 32`.

---

# 7. Energy-gate ablation — ключевой успех

Получили:

```text
Energy gate:

patchwork      Jstruct = 0.666
fractal        Jstruct = 0.363
wavy_stripes   Jstruct = 0.147
noise          Jstruct = 0.0033
checkerboard   Jstruct = 0
```

Retained raw detail energy:

```text
patchwork       0.962
fractal         0.330
wavy            0.222
noise           0.0049
```

Это очень хороший qualitative behavior:

$$
fractal>wavy\gg noise,
$$

и patchwork остается сложным через heterogeneity.

Для patchwork:

$$
J_{\rm nested}=0,
$$

$$
J_{\rm hetero}\approx J_{\rm struct}.
$$

То есть локальные patterns простые, но spatial structural types сильно различаются.

Это сейчас главный candidate gate.

---

# 8. CRH decomposition — текущая ключевая формула

После energy gate заметили очень естественную multiplicative decomposition.

Raw mean detail energy:

$$
\bar E
=
\frac1{N_{\rm pix}}
\sum_{x,k,\alpha}E_{k,\alpha}(x).
$$

В текущих conventions

$$
\bar E \simeq C_{\rm detail}.
$$

Организованная масса:

$$
\bar W
=
\frac1{N_{\rm pix}}
\sum W.
$$

Определяем retained organized-energy fraction:

$$
\boxed{
R=\frac{\bar W}{\bar E}.
}
$$

Тогда:

$$
\bar W=C_{\rm detail}R
$$

и

$$
\boxed{
J_{\rm struct}
=
C_{\rm detail}\;R\;H_{\rm struct}.
}
$$

Это условно **CRH decomposition**:

* \(C=C_{\rm detail}\): сколько detail вообще есть;
* \(R\): какая fraction detail spatially organized;
* \(H=H_{\rm struct}\): насколько разнообразна структура organized detail.

А ветви:

$$
J_{\rm nested}
=
C_{\rm detail}R H_{\rm nested},
$$

$$
J_{\rm hetero}
=
C_{\rm detail}R I_{\rm hetero},
$$

$$
H_{\rm struct}
=
H_{\rm nested}+I_{\rm hetero}.
$$

Очень важный принцип: **мы не хотим additive**

$$
C_{\rm detail}+(\text{structural term}),
$$

потому что тогда high-variance noise/checkerboard автоматически получает complexity.

Нужна именно multiplicative logic:

$$
\text{detail}
\times
\text{organization}
\times
\text{diversity}.
$$

---

# 9. Synthetic validation после energy gate

Последняя clean validation panel дала примерно:

```text
stripes          Jstruct = 0
checkerboard     Jstruct = 0
patchwork        Jstruct = 0.666
nested_dyadic    Jstruct = 0
wavy_stripes     Jstruct = 0.147
fractal          Jstruct ≈ 0.186
noise            Jstruct ≈ 0.002 ± 0.001
```

В целом hierarchy разумная.

Два unresolved points:

### nested_dyadic = 0

Это отдельный sanity failure. Хотелось бы ненулевую complexity.

Пока не разобран.

### fractal vs wavy

После energy gate ordering правильный, хотя gap не огромный.

---

# 10. Natural images и проблема бинаризации

На natural images обнаружился устойчивый эффект:

```text
original -> binary
```

часто увеличивает `Jstruct` в несколько раз.

Примеры:

```text
face       0.016 -> 0.110
fractal    0.061 -> 0.178
leaf       0.029 -> 0.240
```

Это выглядит подозрительно, потому что binarization удаляет:

* grayscale information;
* texture;
* smooth transitions;

но делает edges очень резкими.

Чтобы понять механизм, сделали CRH decomposition.

---

# 11. Binarization diagnostic

Считали:

$$
J_{\rm struct}
=
C_{\rm detail}R H_{\rm struct}.
$$

И diagnostic:

$$
J_{\rm specific}
=
\frac{J_{\rm struct}}{C_{\rm detail}}
=
R H_{\rm struct}.
$$

`Jspecific` пока **не final metric**, только способ отделить amplitude detail от structural part.

### Face

Original:

```text
Cdetail   = 0.093
R         = 0.075
Hstruct   = 2.265
Jstruct   = 0.016
Jspecific = 0.169
```

Binary:

```text
Cdetail   = 0.202
R         = 0.217
Hstruct   = 2.514
Jstruct   = 0.110
Jspecific = 0.546
```

Ratios примерно:

```text
C       ×2.18
R       ×2.9
H       ×1.11
J       ×7
```

То есть здесь бинаризация повышает не только detail amplitude, но и organization fraction \(R\).

---

### Fractal image

Original:

```text
Cdetail   = 0.165
R         = 0.129
Hstruct   = 2.872
Jstruct   = 0.061
Jspecific = 0.371
```

Binary:

```text
Cdetail   = 0.408
R         = 0.150
Hstruct   = 2.918
Jstruct   = 0.178
Jspecific = 0.437
```

Ratios:

```text
C       ×2.47
R       ×1.16
H       ×1.02
J       ×2.92
```

Тут эффект почти целиком из \(C_{\rm detail}\).

---

### Leaf / fern

Original:

```text
Cdetail   = 0.058
R         = 0.185
Hstruct   = 2.662
Jstruct   = 0.029
Jspecific = 0.492
```

Binary:

```text
Cdetail   = 0.362
R         = 0.240
Hstruct   = 2.764
Jstruct   = 0.240
Jspecific = 0.663
```

Ratios примерно:

```text
C       ×6.2
R       ×1.3
H       ×1.04
J       ×8.3
```

Опять main culprit — \(C_{\rm detail}\).

---

# 12. Главный вывод из binarization test

Очень стабильна именно entropy part:

$$
H_{\rm struct}^{binary}
\approx
H_{\rm struct}^{original}.
$$

То есть binarization почти не делает RG-history distribution «в несколько раз богаче».

Большой рост `Jstruct` в основном возникает через prefactor:

$$
C_{\rm detail}R.
$$

Для fractal/leaf главным фактором является \(C_{\rm detail}\): thresholding превращает smooth gradients в full-contrast boundaries.

Face — смешанный случай: растет и \(C\), и сильно \(R\).

То есть structural entropy выглядит гораздо более устойчивой, чем absolute extensive complexity.

---

# 13. Что НЕ хотим делать

Пользователь явно не любит вариант

$$
C_{\rm detail}+J_{\rm specific},
$$

потому что high variance сама по себе тогда дает positive complexity:

* noise;
* checkerboard;
* прочие trivial high-contrast patterns.

Хотим сохранять **multiplicative suppression**.

Пока НЕ вводили новый final scalar.

В частности пока не решили:

* оставить ли linear \(C_{\rm detail}\);
* использовать \(C^\gamma\), \(0<\gamma<1\);
* saturation;
* вовсе сделать contrast-independent variant.

Это следующий conceptual decision, а не уже принятое решение.

---

# 14. Следующий разумный diagnostic

Перед тем как менять amplitude factor, полезно проверить **чистое contrast scaling без thresholding**:

$$
f\rightarrow af.
$$

Для текущей конструкции ожидается:

$$
E\rightarrow a^2E,
$$

$$
C_{\rm detail}\rightarrow a^2C_{\rm detail}.
$$

Pearson `Qenergy` должен быть invariant к positive global amplitude scaling, значит:

$$
R\approx const,
$$

$$
H_{\rm struct}\approx const,
$$

$$
J_{\rm struct}\propto a^2.
$$

Если это подтвердится, станет совсем ясно, что \(C_{\rm detail}\) — отдельная amplitude axis, а structural shape лежит в

$$
R H_{\rm struct}.
$$

После этого уже решать, насколько amplitude должна входить в final complexity.

---

# 15. Phase decomposition — вторичная ветка, пока отложена

До смены gate у нас была дополнительная decomposition nested branch:

$$
J_{\rm nested}
=
J_{\rm spectral}
+
J_{\rm phase}.
$$

Где

$$
J_{\rm spectral}
=
\langle
J_{\rm nested}^{phase\ scrambled}
\rangle,
$$

$$
J_{\rm phase}
=
J_{\rm nested}^{original}
-
J_{\rm spectral}.
$$

`Jphase` signed:

* \(>0\): original phases add organization;
* \(\approx0\): nested richness largely spectrum-conditioned;
* \(<0\): special phase locking makes pattern simpler than typical phase surrogate.

Notable old cases:

* checkerboard: strongly negative;
* wavy stripes: near zero;
* nested dyadic: positive;
* fractal: near zero.

Но это всё тестировалось в основном на старой `qorient`-версии.

**Phase decomposition надо в будущем перепроверить с новым `Qenergy` gate**, прежде чем включать в paper core.

Пока current core лучше мыслить как:

$$
J_{\rm struct}
=
C R
\left(
H_{\rm nested}+I_{\rm hetero}
\right).
$$

---

# 16. Legacy metric zoo

Старые quantities оставить в code diagnostics, но не держать в голове как final:

* \(Q_k\): old orientation coherence;
* \(q_k(x)\): local orientation coherence;
* \(D_k\): orientation diversity;
* \(O_k=C_kQ_k\);
* `Odiv`;
* `Jglobal`;
* old `Jloc`;
* `JlocQ` = исторический ancestor нынешнего `Jnested`.

Главная проблема old `Jglobal`: global diversity без locality → patchwork выглядел «слишком сложным».

Теперь это переосмыслено: spatial heterogeneity сама по себе legitimate branch, но должна выделяться отдельно через \(I(X;C)\).

---

# 17. Repo / scripts, которые использовались

Основные актуальные diagnostics примерно такие:

```text
scripts/benchmark_complexity_tree.py
scripts/diagnose_q_and_support.py
scripts/validate_mvp_complexity.py
scripts/diagnose_binarization.py
```

Codex в этой сессии почему-то часто не мог запускать scripts из своего environment, поэтому последние спеки специально формулировались как:

> inspect repo + write code only; do not run.

Пользователь запускает scripts вручную.

---

# 18. Текущий conceptual picture

Самая компактная версия:

$$
\boxed{
J_{\rm struct}
=
C_{\rm detail}
\times
R
\times
H_{\rm struct}
}
$$

где

$$
H_{\rm struct}
=
H_{\rm nested}
+
I_{\rm hetero}.
$$

То есть structural complexity требует одновременно:

1. **detail exists**;
2. **detail is spatially organized rather than noise**;
3. **organized detail has nontrivial structural diversity**.

И diversity имеет два вида:

* nested / multiscale richness within local RG histories;
* heterogeneity between local histories across space.

Это сейчас выглядит как наиболее чистая основа для MVP definition.

---

# 19. Основные unresolved вопросы

В порядке важности:

1. **Amplitude dependence / binarization**

   * linear \(C_{\rm detail}\) делает binary natural images сильно более complex;
   * Hstruct при этом устойчив;
   * нужно понять, как именно amplitude должна входить в final metric.

2. **Nested dyadic = 0**

   * synthetic hierarchy в целом хорошая;
   * но hierarchical dyadic example получает ноль, что выглядит подозрительно.

3. **Qenergy edge cases**

   * constant native energy → Pearson undefined;
   * very coarse scales → too few neighbor pairs;
   * сейчас есть min-pair safeguard.

4. **Revalidate phase branch**

   * `Jspectral/Jphase` нужно пересчитать после перехода от qorient к Qenergy.

5. После этого **freeze metric design** и перейти к physical MVP:

   * 2D Ising;
   * возможно DM/labyrinth/skyrmion patterns.

---

# 20. Что я бы делал первым в новом чате

Самый естественный следующий шаг:

**contrast-scaling diagnostic**

для нескольких fixed images:

$$
f_a=a f
$$

с несколькими \(a\), не делая clipping/thresholding там, где это возможно.

Проверить:

```text
Cdetail(a)
R(a)
Hstruct(a)
Jstruct(a)
Jspecific(a)=R*Hstruct
```

Если:

```text
Cdetail ∝ a^2
R ≈ const
Hstruct ≈ const
```

то amplitude dependence будет полностью изолирована, и можно уже предметно решать, нужен ли:

$$
C,
\quad
C^\gamma,
\quad
\text{saturating }C,
\quad
\text{или contrast-independent structural complexity}.
$$

При этом additive decomposition не рассматриваем: хотим сохранить multiplicative suppression шума и trivial order.

---
