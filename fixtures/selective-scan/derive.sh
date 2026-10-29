#!/usr/bin/env bash
# Independent expected values for tests/selective_scan_tests.cpp. Every value is
# derived with GNU bc (arbitrary precision) from a closed form or an explicit
# double sum over the contract formula, never from the C++ recurrence. Case
# inputs are defined in README.md. Output rows: case, index, value, exp2, where
# expected = value * 2^exp2 and value is truncated to 40 decimals.
#   bash fixtures/selective-scan/derive.sh > fixtures/selective-scan/expected.tsv
set -euo pipefail
export BC_LINE_LENGTH=0
bc -lq <<'BC'
scale = 80
ln2 = l(2)
define sp(x) { if (x > 20) return (x); return (l(1 + e(x))); }
define md(a, m) { auto s, r; s = scale; scale = 0; r = a % m; scale = s; return (r); }
define p(v) { auto s; s = scale; scale = 40; v = v / 1; print v; scale = s; return (0); }
define gs(a, t) { if (a == 1) return (t + 1); return ((1 - a ^ (t + 1)) / (1 - a)); }

/* single_token: Delta = ln2 (delta 0) and 21 (passthrough); hand form 1 + 4 ln2, -164 */
print "single_token\t0\t"; z = p(1 + 4 * ln2); print "\t0\n"
print "single_token\t1\t"; z = p(-164); print "\t0\n"

/* three_token_decay: decay 1/2, u = s + 1, y_t = ln2 * sum_{s<=t} (1/2)^(t-s) u_s */
for (t = 0; t < 3; t++) {
  v = 0; for (j = 0; j <= t; j++) v += (1 / 2) ^ (t - j) * (j + 1)
  print "three_token_decay\t", t, "\t"; z = p(v * ln2); print "\t0\n"
}

/* softplus_branches: y_d = softplus(delta_d) */
x[0] = -30; x[1] = -20; x[2] = -1; x[3] = 0; x[4] = 20; x[5] = 20.5
for (d = 0; d < 6; d++) { print "softplus_branches\t", d, "\t"; z = p(sp(x[d])); print "\t0\n" }

/* geometric_p3_length: y[t,d] = u_d (ln2 G_t + 1/2), a_n = 2^-(n+1) */
for (t = 0; t < 400; t++) {
  g = 0; for (n = 0; n < 16; n++) g += gs(2 ^ (-(n + 1)), t)
  for (d = 0; d < 4; d++) {
    print "geometric_p3_length\t", t * 4 + d, "\t"; z = p((d + 1) / 4 * (ln2 * g + 1 / 2)); print "\t0\n"
  }
}

/* per_channel_a: a_{d,n} = 2^A_{d,n} */
a[0] = -1; a[1] = -2; a[2] = -.5; a[3] = -4; a[4] = 0; a[5] = -1
b[0] = 1; b[1] = -.5; c[0] = 2; c[1] = 1
u[0] = 1; u[1] = -2; u[2] = .5; q[0] = 0; q[1] = .25; q[2] = -1
for (t = 0; t < 25; t++) for (d = 0; d < 3; d++) {
  v = q[d] * u[d]
  for (n = 0; n < 2; n++) { r = 1; if (a[d * 2 + n] != 0) r = e(a[d * 2 + n] * ln2); v += c[n] * ln2 * b[n] * u[d] * gs(r, t) }
  print "per_channel_a\t", t * 3 + d, "\t"; z = p(v); print "\t0\n"
}

/* state_32_p4_length: A_n = -(n+1)/8, C = 1/2, u = (1, -1), no D */
for (t = 0; t < 100; t++) {
  g = 0; for (n = 0; n < 32; n++) g += gs(e(-(n + 1) / 8 * ln2), t)
  for (d = 0; d < 2; d++) { print "state_32_p4_length\t", t * 2 + d, "\t"; z = p((1 - 2 * d) / 2 * ln2 * g); print "\t0\n" }
}

/* varying_direct_sum: explicit double sum, no recurrence:
   y = sum_n C_tn sum_{s<=t} exp(A_n sum_{s<r<=t} Delta_r) Delta_s B_sn u_s + D u_t */
define dl(b, t, d) { return ((md(7 * b + 3 * t + 5 * d, 13) - 6) / 2); }
define uu(b, t, d) { return ((md(5 * b + 7 * t + 3 * d, 11) - 5) / 4); }
define bb(b, t, n) { return ((md(3 * b + 5 * t + 7 * n, 9) - 4) / 4); }
define cc(b, t, n) { return ((md(7 * b + 3 * t + 5 * n, 7) - 3) / 2); }
w[0] = -.5; w[1] = -1; w[2] = -2; w[3] = -3
for (i = 0; i < 2; i++) for (t = 0; t < 7; t++) for (d = 0; d < 3; d++) k[(i * 7 + t) * 3 + d] = sp(dl(i, t, d))
for (i = 0; i < 2; i++) for (t = 0; t < 7; t++) for (d = 0; d < 3; d++) {
  v = (d - 1) / 2 * uu(i, t, d)
  for (n = 0; n < 4; n++) for (j = 0; j <= t; j++) {
    m = 0; for (r = j + 1; r <= t; r++) m += k[(i * 7 + r) * 3 + d]
    v += cc(i, t, n) * e(w[n] * m) * k[(i * 7 + j) * 3 + d] * bb(i, j, n) * uu(i, j, d)
  }
  print "varying_direct_sum\t", (i * 7 + t) * 3 + d, "\t"; z = p(v); print "\t0\n"
}

/* m_head_k: K_t = ln2 sum_n B_n (1 - 2^-(n+1)(t+1)) / (1 - 2^-(n+1)), B_n = (n mod 4 + 1)/4 */
for (t = 0; t < 400; t++) {
  g = 0; for (n = 0; n < 16; n++) g += (md(n, 4) + 1) / 4 * gs(2 ^ (-(n + 1)), t)
  print "m_head_k\t", t, "\t"; z = p(ln2 * g); print "\t0\n"
}

/* extreme cases */
print "extreme_large_delta_memoryless\t0\t3\t0\n"
print "extreme_large_delta_memoryless\t1\t-5\t0\n"
print "extreme_large_delta_memoryless\t2\t7\t0\n"
/* softplus(-2^100) * 2^100 < 10^-(10^29): the D skip alone remains */
print "extreme_negative_delta\t0\t1\t0\n"
print "extreme_negative_delta\t1\t-1\t0\n"
scale = 200
print "extreme_fp32_max_input\t0\t"; z = p(2 ^ 127 * l(1 + e(-100))); print "\t0\n"
scale = 80
print "extreme_near_fp32_limit_output\t0\t"; z = p(21 * 2 ^ 122); print "\t0\n"
print "extreme_subnormal_input\t0\t"; z = p(ln2); print "\t-149\n"
print "extreme_subnormal_input\t1\t"; z = p(3 / 2 * ln2); print "\t-149\n"
for (t = 0; t < 3; t++) { print "extreme_most_negative_a\t", t, "\t"; z = p((t + 1) * ln2); print "\t0\n" }

/* cross_scan_prefix_merge: hand-derived direction prefix sums (README) */
for (j = 0; j < 6; j++) { print "cross_scan_prefix_merge\t", j, "\t"; z = p((22 + j) / 2 * ln2); print "\t0\n" }
BC
