#!/usr/bin/env python3
"""Search for nostr npubs that start with a given bech32 prefix, on an NVIDIA GPU.

Each GPU thread walks a run of consecutive private keys using affine point
addition with batched inversion. Every point P also gives beta*P.x and
beta^2*P.x through the secp256k1 endomorphism (keys lambda*k and lambda^2*k),
so one point checks 3 npubs. The host recomputes every hit from scratch before
reporting it.

Usage: vanity_npub.py PREFIX [--out DIR]
The nsec goes to a 0600 file in --out. Only the npub is printed.
"""
import argparse, ctypes, glob, os, secrets, sys, time

import ptxgen
from ctypes import c_int, c_uint, c_uint64, c_size_t, c_void_p, c_char_p, byref, POINTER

# secp256k1, used on the host for setup and to verify hits
P = 2**256 - 2**32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
     0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
BETA = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
LAMBDA = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72


def ec_add(a, b):
    if a is None: return b
    if b is None: return a
    if a[0] == b[0]:
        if (a[1] + b[1]) % P == 0: return None
        lam = 3 * a[0] * a[0] * pow(2 * a[1], -1, P) % P
    else:
        lam = (b[1] - a[1]) * pow(b[0] - a[0], -1, P) % P
    x = (lam * lam - a[0] - b[0]) % P
    return (x, (lam * (a[0] - x) - a[1]) % P)


def ec_mul(k, pt=G):
    r = None
    while k:
        if k & 1: r = ec_add(r, pt)
        pt = ec_add(pt, pt)
        k >>= 1
    return r

# bech32 (BIP-173)
CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _polymod(values):
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if (b >> i) & 1 else 0
    return chk


def bech32_encode(hrp, data8):
    acc = bits = 0
    data = []
    for b in data8:
        acc = (acc << 8) | b
        bits += 8
        while bits >= 5:
            bits -= 5
            data.append((acc >> bits) & 31)
    if bits: data.append((acc << (5 - bits)) & 31)
    hx = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    pm = _polymod(hx + data + [0] * 6) ^ 1
    data += [(pm >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(CHARSET[d] for d in data)


def npub_of(k):
    return bech32_encode("npub", ec_mul(k)[0].to_bytes(32, "big"))

# Plain C field arithmetic, kept for comparison with the PTX version (--cfield).
C_FIELD = r"""
DEV void fe_reduce(u32 r[8], const u32 t[16]) {
  // hi * 2^256 == hi * (2^32 + 977)
  u32 s[8];
  u64 c = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    c += (u64)t[i] + (u64)t[8 + i] * 977u;
    if (i > 0) c += t[7 + i];
    s[i] = (u32)c; c >>= 32;
  }
  c += t[15];
  u64 lo = c * 977u;
  u64 acc = (u64)s[0] + (u32)lo; r[0] = (u32)acc; acc >>= 32;
  acc += (u64)s[1] + (lo >> 32) + (u32)c; r[1] = (u32)acc; acc >>= 32;
  acc += (u64)s[2] + (c >> 32); r[2] = (u32)acc; acc >>= 32;
#pragma unroll
  for (int i = 3; i < 8; i++) { acc += s[i]; r[i] = (u32)acc; acc >>= 32; }
  if (acc) fe_add_c(r);
  fe_norm(r);
}

DEV void fe_mul(u32 r[8], const u32 a[8], const u32 b[8]) {
  u32 t[16];
#pragma unroll
  for (int i = 0; i < 16; i++) t[i] = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    u64 c = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) { c += (u64)a[i] * b[j] + t[i + j]; t[i + j] = (u32)c; c >>= 32; }
    t[i + 8] = (u32)c;
  }
  fe_reduce(r, t);
}

DEV void fe_sqr(u32 r[8], const u32 a[8]) {
  u32 t[16];
#pragma unroll
  for (int i = 0; i < 16; i++) t[i] = 0;
#pragma unroll
  for (int i = 0; i < 7; i++) {  // cross terms a[i]*a[j], i < j
    u64 c = 0;
#pragma unroll
    for (int j = i + 1; j < 8; j++) { c += (u64)a[i] * a[j] + t[i + j]; t[i + j] = (u32)c; c >>= 32; }
    t[i + 8] = (u32)c;
  }
  t[15] = t[14] >> 31;
#pragma unroll
  for (int i = 14; i > 0; i--) t[i] = (t[i] << 1) | (t[i - 1] >> 31);
  t[0] <<= 1;
  u64 c = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {  // + diagonal a[i]^2
    u64 sq = (u64)a[i] * a[i];
    c += (u64)t[2 * i] + (u32)sq; t[2 * i] = (u32)c; c >>= 32;
    c += (u64)t[2 * i + 1] + (sq >> 32); t[2 * i + 1] = (u32)c; c >>= 32;
  }
  fe_reduce(r, t);
}

"""

KERNEL = r"""
typedef unsigned int u32;
typedef unsigned long long u64;

__constant__ u32 TGX[(BATCH + 1) * 8];
__constant__ u32 TGY[(BATCH + 1) * 8];
__constant__ u32 CBETA[8];
__constant__ u32 CBETA2[8];
__constant__ u32 ZERO8[8];

#define DEV __device__ __forceinline__

DEV void fe_add_c(u32 r[8]) {  // r += 2^256 - p  (i.e. 0x1000003D1)
  u64 c = (u64)r[0] + 0x3D1u; r[0] = (u32)c; c >>= 32;
  c += (u64)r[1] + 1u; r[1] = (u32)c; c >>= 32;
#pragma unroll
  for (int i = 2; i < 8; i++) { c += r[i]; r[i] = (u32)c; c >>= 32; }
}

DEV void fe_norm(u32 r[8]) {  // r < 2^256 -> r < p
  if ((r[7] & r[6] & r[5] & r[4] & r[3] & r[2]) == 0xFFFFFFFFu &&
      (r[1] > 0xFFFFFFFEu || (r[1] == 0xFFFFFFFEu && r[0] >= 0xFFFFFC2Fu)))
    fe_add_c(r);
}

DEV void fe_add(u32 r[8], const u32 a[8], const u32 b[8]) {
  u64 c = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) { c += (u64)a[i] + b[i]; r[i] = (u32)c; c >>= 32; }
  if (c) fe_add_c(r);
  fe_norm(r);
}

DEV void fe_sub(u32 r[8], const u32 a[8], const u32 b[8]) {
  u32 bw = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) { u64 d = (u64)a[i] - b[i] - bw; r[i] = (u32)d; bw = (u32)(d >> 63); }
  if (bw) {  // r += p  ==  r -= 0x1000003D1 (mod 2^256)
    u64 d = (u64)r[0] - 0x3D1u; r[0] = (u32)d; bw = (u32)(d >> 63);
    d = (u64)r[1] - 1u - bw; r[1] = (u32)d; bw = (u32)(d >> 63);
#pragma unroll
    for (int i = 2; i < 8; i++) { d = (u64)r[i] - bw; r[i] = (u32)d; bw = (u32)(d >> 63); }
  }
}

@@FIELD@@
__device__ __noinline__ void fe_sqrn(u32 r[8], const u32 a[8], int n) {
  fe_sqr(r, a);
#pragma unroll 1
  for (int i = 1; i < n; i++) fe_sqr(r, r);
}

DEV void fe_sqrn_mul(u32 r[8], const u32 a[8], int n, const u32 m[8]) {
  u32 t[8]; fe_sqrn(t, a, n); fe_mul(r, t, m);
}

DEV void fe_inv(u32 r[8], const u32 a[8]) {  // a^(p-2), libsecp256k1 addition chain
  u32 x2[8], x3[8], x6[8], x9[8], x11[8], x22[8], x44[8], x88[8], x176[8], x220[8], x223[8], t[8];
  fe_sqrn_mul(x2, a, 1, a);
  fe_sqrn_mul(x3, x2, 1, a);
  fe_sqrn_mul(x6, x3, 3, x3);
  fe_sqrn_mul(x9, x6, 3, x3);
  fe_sqrn_mul(x11, x9, 2, x2);
  fe_sqrn_mul(x22, x11, 11, x11);
  fe_sqrn_mul(x44, x22, 22, x22);
  fe_sqrn_mul(x88, x44, 44, x44);
  fe_sqrn_mul(x176, x88, 88, x88);
  fe_sqrn_mul(x220, x176, 44, x44);
  fe_sqrn_mul(x223, x220, 3, x3);
  fe_sqrn_mul(t, x223, 23, x22);
  fe_sqrn_mul(t, t, 5, a);
  fe_sqrn_mul(t, t, 3, x2);
  fe_sqrn_mul(r, t, 2, a);
}

struct Ctx { u64 mask, target; u32 *out; u32 tid, round, orb; };

DEV void record(const Ctx &cx, int j, u32 e) {
  u32 i = atomicAdd(cx.out, 1u);
  if (i < MAXOUT) {
    u32 *o = cx.out + 1 + i * 4;
    o[0] = cx.tid; o[1] = cx.round; o[2] = (u32)j; o[3] = e + 3u * cx.orb;
  }
}

DEV bool match(const u32 x[8], const Ctx &cx) {
  return ((((u64)x[7] << 32) | x[6]) & cx.mask) == cx.target;
}

// Estimate the top 32 bits of (K*x mod p) for a constant K from only the product
// columns that feed limb 7. (true - estimate) mod 2^32 lies in [0, FILTER_TOL)
// unless a carry lands exactly on a limb boundary. Candidates that pass are
// checked exactly, so a bad estimate can only cause a missed match, never a
// wrong one. --audit counts how often the bound is violated.
#define FILTER_TOL 64u
DEV u32 top_est(const u32 a[8], const u32 *K) {
  // P7 ~ lo32(sum lo(col 7) + sum hi(col 6)); carry from col <= 6 ignored (<= 14)
  u64 c7 = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) c7 += a[i] * K[7 - i];
#pragma unroll
  for (int i = 0; i < 7; i++) c7 += __umulhi(a[i], K[6 - i]);
  u32 p7 = (u32)c7;
  // P12..P15: full columns 12..14 plus hi(col 11); carry into col 12 ignored
  u64 c = 0;
#pragma unroll
  for (int i = 4; i < 8; i++) c += __umulhi(a[i], K[11 - i]);
#pragma unroll
  for (int i = 5; i < 8; i++) c += a[i] * K[12 - i];
  c >>= 32;  // -> col 13
#pragma unroll
  for (int i = 5; i < 8; i++) c += __umulhi(a[i], K[12 - i]);
#pragma unroll
  for (int i = 6; i < 8; i++) c += a[i] * K[13 - i];
  c >>= 32;  // -> col 14
#pragma unroll
  for (int i = 6; i < 8; i++) c += __umulhi(a[i], K[13 - i]);
  c += a[7] * K[7];
  u32 p14 = (u32)c;
  c >>= 32;
  c += __umulhi(a[7], K[7]);
  u32 p15 = (u32)c;
  // r7 = P7 + P14 + lo32(977*P15) + hi32(977*P14)  (from hi*2^256 == hi*(2^32+977))
  return p7 + p14 + p15 * 977u + __umulhi(p14, 977u);
}

DEV bool may_match(u32 est, const Ctx &cx, u32 tol) {
  u32 m = (u32)(cx.mask >> 32), t = (u32)(cx.target >> 32);
  return (t - est) < tol || (est & m) == t || ((est + tol - 1) & m) == t;
}

// beta^2*x = -x - beta*x (mod p), so its top limb is -1 - x7 - top(beta*x) - {0,1,2}.
// Derive its estimate from the beta*x estimate, with a slightly wider tolerance.
#define FILTER_TOL2 (FILTER_TOL + 4u)
DEV u32 top_est2(const u32 x[8], u32 e1) { return (0u - 1u - x[7] - e1) - (FILTER_TOL2 - 1u); }

DEV void check(const u32 x[8], const Ctx &cx, int j) {
  if (match(x, cx)) record(cx, j, 0);
  u32 e1 = top_est(x, CBETA), e2 = top_est2(x, e1);
#ifdef AUDIT
  { u32 y[8], b[8];
#pragma unroll
    for (int i = 0; i < 8; i++) b[i] = CBETA[i];
    fe_mul(y, x, b);
    if (y[7] - e1 >= FILTER_TOL) record(cx, 0x40000001, 1);
    if (match(y, cx)) record(cx, may_match(e1, cx, FILTER_TOL) ? j : 0x40000000, 1);
    fe_mul(y, y, b);
    if (y[7] - e2 >= FILTER_TOL2) record(cx, 0x40000001, 2);
    if (match(y, cx)) record(cx, may_match(e2, cx, FILTER_TOL2) ? j : 0x40000000, 2);
  }
#else
  if (may_match(e1, cx, FILTER_TOL)) {
    u32 y[8], b[8];
#pragma unroll
    for (int i = 0; i < 8; i++) b[i] = CBETA[i];
    fe_mul(y, x, b);
    if (match(y, cx)) record(cx, j, 1);
  }
  if (may_match(e2, cx, FILTER_TOL2)) {
    u32 y[8], b[8];
#pragma unroll
    for (int i = 0; i < 8; i++) b[i] = CBETA2[i];
    fe_mul(y, x, b);
    if (match(y, cx)) record(cx, j, 2);
  }
#endif
}

#ifdef ORBIT
// Orbit kernel. Table entry k holds Q = (k+1)G = (a, b), whose endomorphism images
// are phi^i(Q) = (beta^i a, b) = lambda^i Q. With z_i = beta^i a and c = cx, the
// denominators d_i = z_i - c multiply to e = a^3 - c^3, and 1/d_i = m_i / e where
// m_i = z_i^2 + z_i c + c^2. So one batch-inversion element (e) covers the 6 points
// C + eps*phi^i(Q), and (eps*b - cy)/e is shared by all three values of i.
// Rows: 0 a, 1 beta a, 2 beta^2 a, 3 a^2, 4 beta^2 a^2, 5 beta a^2, 6 a^3, 7 b.
__constant__ u32 OT[ORBIT * 8 * 8];
__constant__ u32 OJ[16];  // jump point (2*ORBIT+1)G: x, y

#define OROW(k, r) (OT + ((k) * 8 + (r)) * 8)
DEV void ld(u32 d[8], const u32 *s) {
#pragma unroll
  for (int i = 0; i < 8; i++) d[i] = s[i];
}

extern "C" __global__ void __launch_bounds__(TPB, MINB)
search_orbit(u32 *gcx, u32 *gcy, u32 nthreads, u32 rounds, u64 mask, u64 target, u32 *out) {
  u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nthreads) return;
  Ctx ctx; ctx.mask = mask; ctx.target = target; ctx.out = out; ctx.tid = tid; ctx.orb = 0;
  u32 cx[8], cy[8];
#pragma unroll
  for (int k = 0; k < 8; k++) { cx[k] = gcx[k * nthreads + tid]; cy[k] = gcy[k * nthreads + tid]; }
  u32 prod[(ORBIT + 1) * 8];

#pragma unroll 1
  for (u32 r = 0; r < rounds; r++) {
    ctx.round = r;
    u32 c2[8], c3[8], d[8], acc[8], g[8];
    fe_sqr(c2, cx);
    fe_mul(c3, c2, cx);
#pragma unroll 1
    for (int k = 0; k < ORBIT; k++) {
      ld(g, OROW(k, 6));
      fe_sub(d, g, c3);
      if (k == 0) {
#pragma unroll
        for (int q = 0; q < 8; q++) acc[q] = d[q];
      } else fe_mul(acc, acc, d);
#pragma unroll
      for (int q = 0; q < 8; q++) prod[k * 8 + q] = acc[q];
    }
    ld(g, OJ);
    fe_sub(d, g, cx);
    fe_mul(acc, acc, d);
    u32 inv[8];
    fe_inv(inv, acc);
    // jump element: next center
    u32 nx[8], ny[8], ii[8], lam[8], t[8], x[8], h[8];
    {
      u32 pp[8];
#pragma unroll
      for (int q = 0; q < 8; q++) pp[q] = prod[(ORBIT - 1) * 8 + q];
      fe_mul(ii, inv, pp);
      fe_mul(inv, inv, d);
      ld(h, OJ + 8);
      fe_sub(t, h, cy); fe_mul(lam, t, ii);
      fe_sqr(nx, lam); fe_sub(nx, nx, cx); fe_sub(nx, nx, g);
      fe_sub(t, cx, nx); fe_mul(t, lam, t); fe_sub(ny, t, cy);
    }
#pragma unroll 1
    for (int k = ORBIT - 1; k >= 0; k--) {
      if (k > 0) {
        u32 pp[8];
#pragma unroll
        for (int q = 0; q < 8; q++) pp[q] = prod[(k - 1) * 8 + q];
        fe_mul(ii, inv, pp);
        ld(g, OROW(k, 6));
        fe_sub(d, g, c3);
        fe_mul(inv, inv, d);
      } else {
#pragma unroll
        for (int q = 0; q < 8; q++) ii[q] = inv[q];
      }
      // ii = 1/e. wp = (b - cy)/e and wm = (b + cy)/e; the sign of the slope
      // doesn't matter once it's squared.
      u32 wp[8], wm[8], u0[8], u1[8], u2[8];
      ld(h, OROW(k, 7));
      fe_sub(t, h, cy); fe_mul(wp, t, ii);
      fe_add(t, h, cy); fe_mul(wm, t, ii);
      ld(g, OROW(k, 0)); fe_mul(u0, g, cx);
      ld(g, OROW(k, 1)); fe_mul(u1, g, cx);
      fe_add(u2, u0, u1); fe_sub(u2, ZERO8, u2);
#pragma unroll
      for (int i = 0; i < 3; i++) {
        u32 m[8];
        ld(g, OROW(k, 3 + i));
        fe_add(m, g, i == 0 ? u0 : (i == 1 ? u1 : u2));
        fe_add(m, m, c2);
        ld(g, OROW(k, i));  // z_i
        ctx.orb = i;
        fe_mul(lam, wp, m);
        fe_sqr(x, lam); fe_sub(x, x, cx); fe_sub(x, x, g);
        check(x, ctx, k + 1);
        fe_mul(lam, wm, m);
        fe_sqr(x, lam); fe_sub(x, x, cx); fe_sub(x, x, g);
        check(x, ctx, -(k + 1));
      }
      ctx.orb = 0;
    }
    check(cx, ctx, 0);
#pragma unroll
    for (int q = 0; q < 8; q++) { cx[q] = nx[q]; cy[q] = ny[q]; }
  }
#pragma unroll
  for (int q = 0; q < 8; q++) { gcx[q * nthreads + tid] = cx[q]; gcy[q * nthreads + tid] = cy[q]; }
}
#endif

extern "C" __global__ void unit(const u32 *in, u32 *out, u32 n) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 a[8], b[8], r[8], q[8];
#pragma unroll
  for (int k = 0; k < 8; k++) { a[k] = in[i * 16 + k]; b[k] = in[i * 16 + 8 + k]; }
  fe_mul(r, a, b);
  fe_sqr(q, a);
#pragma unroll
  for (int k = 0; k < 8; k++) { out[i * 16 + k] = r[k]; out[i * 16 + 8 + k] = q[k]; }
}

// Each round, center C covers keys c-BATCH..c+BATCH, then C += (2*BATCH+1)G.
// Table index i < BATCH holds (i+1)G; index BATCH holds (2*BATCH+1)G.
extern "C" __global__ void __launch_bounds__(TPB, MINB)
search(u32 *gcx, u32 *gcy, u32 nthreads, u32 rounds, u64 mask, u64 target, u32 *out) {
  u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nthreads) return;
  Ctx ctx; ctx.mask = mask; ctx.target = target; ctx.out = out; ctx.tid = tid; ctx.orb = 0;
  u32 cx[8], cy[8];
#pragma unroll
  for (int k = 0; k < 8; k++) { cx[k] = gcx[k * nthreads + tid]; cy[k] = gcy[k * nthreads + tid]; }
  u32 prod[(BATCH + 1) * 8];

#pragma unroll 1
  for (u32 r = 0; r < rounds; r++) {
    ctx.round = r;
    u32 d[8], acc[8], g[8], h[8];
#pragma unroll 1
    for (int i = 0; i <= BATCH; i++) {
#pragma unroll
      for (int k = 0; k < 8; k++) g[k] = TGX[i * 8 + k];
      fe_sub(d, g, cx);
      if (i == 0) {
#pragma unroll
        for (int k = 0; k < 8; k++) acc[k] = d[k];
      } else fe_mul(acc, acc, d);
#pragma unroll
      for (int k = 0; k < 8; k++) prod[i * 8 + k] = acc[k];
    }
    u32 inv[8];
    fe_inv(inv, acc);
    u32 nx[8], ny[8];
#pragma unroll 1
    for (int i = BATCH; i >= 0; i--) {
#pragma unroll
      for (int k = 0; k < 8; k++) { g[k] = TGX[i * 8 + k]; h[k] = TGY[i * 8 + k]; }
      u32 ii[8];
      if (i > 0) {
        u32 pp[8];
#pragma unroll
        for (int k = 0; k < 8; k++) pp[k] = prod[(i - 1) * 8 + k];
        fe_mul(ii, inv, pp);
        fe_sub(d, g, cx);
        fe_mul(inv, inv, d);
      } else {
#pragma unroll
        for (int k = 0; k < 8; k++) ii[k] = inv[k];
      }
      u32 lam[8], x[8], t[8];
      fe_sub(t, h, cy); fe_mul(lam, t, ii);
      fe_sqr(x, lam); fe_sub(x, x, cx); fe_sub(x, x, g);
      if (i == BATCH) {
#pragma unroll
        for (int k = 0; k < 8; k++) nx[k] = x[k];
        fe_sub(t, cx, x); fe_mul(t, lam, t); fe_sub(ny, t, cy);
      } else {
        check(x, ctx, i + 1);
        // C - G_i: the slope uses -gy, and its square doesn't depend on the sign
        fe_add(t, h, cy); fe_mul(lam, t, ii);
        fe_sqr(x, lam); fe_sub(x, x, cx); fe_sub(x, x, g);
        check(x, ctx, -(i + 1));
      }
    }
    check(cx, ctx, 0);
#pragma unroll
    for (int k = 0; k < 8; k++) { cx[k] = nx[k]; cy[k] = ny[k]; }
  }
#pragma unroll
  for (int k = 0; k < 8; k++) { gcx[k * nthreads + tid] = cx[k]; gcy[k * nthreads + tid] = cy[k]; }
}
"""

# Minimal CUDA driver API and NVRTC bindings


class Cuda:
    def __init__(self):
        self.cu = ctypes.CDLL("libcuda.so.1")
        here = os.path.dirname(os.path.abspath(__file__))
        libs = sorted(glob.glob(os.path.join(here, "deps", "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.*")))
        libs += ["libnvrtc.so"]
        for lib in libs:
            try:
                self.nv = ctypes.CDLL(lib)
                break
            except OSError:
                pass
        else:
            sys.exit("libnvrtc not found. Install the CUDA toolkit or run: "
                     "pip install --target deps nvidia-cuda-nvrtc-cu12")
        self.ck(self.cu.cuInit(0))
        dev = c_int()
        self.ck(self.cu.cuDeviceGet(byref(dev), 0))
        self.dev = dev.value
        ctx = c_void_p()
        self.ck(self.cu.cuDevicePrimaryCtxRetain(byref(ctx), dev))
        self.ck(self.cu.cuCtxSetCurrent(ctx))
        self.cu.cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
                                           c_uint, c_void_p, POINTER(c_void_p), c_void_p]
        self.cu.cuMemcpyHtoD_v2.argtypes = [c_uint64, c_void_p, c_size_t]
        self.cu.cuMemcpyDtoH_v2.argtypes = [c_void_p, c_uint64, c_size_t]
        self.cu.cuMemsetD32_v2.argtypes = [c_uint64, c_uint, c_size_t]
        self.cu.cuModuleGetFunction.argtypes = [POINTER(c_void_p), c_void_p, c_char_p]
        self.cu.cuModuleGetGlobal_v2.argtypes = [POINTER(c_uint64), POINTER(c_size_t), c_void_p, c_char_p]
        self.cu.cuModuleLoadData.argtypes = [POINTER(c_void_p), c_void_p]

    @staticmethod
    def ck(r, what="cuda"):
        if r != 0: raise RuntimeError(f"{what} error {r}")

    def attr(self, a):
        v = c_int()
        self.ck(self.cu.cuDeviceGetAttribute(byref(v), a, self.dev))
        return v.value

    def compile(self, src, opts, verbose=False):
        prog = c_void_p()
        self.ck(self.nv.nvrtcCreateProgram(byref(prog), src.encode(), b"k.cu", 0, None, None), "nvrtc")
        arr = (c_char_p * len(opts))(*[o.encode() for o in opts])
        r = self.nv.nvrtcCompileProgram(prog, len(opts), arr)
        n = c_size_t()
        self.nv.nvrtcGetProgramLogSize(prog, byref(n))
        log = ctypes.create_string_buffer(n.value)
        self.nv.nvrtcGetProgramLog(prog, log)
        if r != 0: sys.exit("nvrtc compile failed:\n" + log.value.decode())
        if verbose: print(log.value.decode().strip(), flush=True)
        self.ck(self.nv.nvrtcGetCUBINSize(prog, byref(n)), "nvrtc")
        cubin = ctypes.create_string_buffer(n.value)
        self.ck(self.nv.nvrtcGetCUBIN(prog, cubin), "nvrtc")
        if os.environ.get("DUMP_CUBIN"):
            open(os.environ["DUMP_CUBIN"], "wb").write(cubin.raw)
        mod = c_void_p()
        self.ck(self.cu.cuModuleLoadData(byref(mod), cubin))
        return mod

    def func(self, mod, name):
        f = c_void_p()
        self.ck(self.cu.cuModuleGetFunction(byref(f), mod, name.encode()))
        return f

    def set_global(self, mod, name, data):
        p, n = c_uint64(), c_size_t()
        self.ck(self.cu.cuModuleGetGlobal_v2(byref(p), byref(n), mod, name.encode()))
        assert n.value == len(data), (name, n.value, len(data))
        self.ck(self.cu.cuMemcpyHtoD_v2(p.value, data, len(data)))

    def alloc(self, n):
        p = c_uint64()
        self.ck(self.cu.cuMemAlloc_v2(byref(p), c_size_t(n)))
        return p.value

    def h2d(self, dst, data): self.ck(self.cu.cuMemcpyHtoD_v2(dst, data, len(data)))

    def d2h(self, src, n):
        buf = ctypes.create_string_buffer(n)
        self.ck(self.cu.cuMemcpyDtoH_v2(buf, src, n))
        return buf.raw

    def launch(self, f, grid, block, args):
        params = (c_void_p * len(args))(*[ctypes.cast(byref(a), c_void_p) for a in args])
        self.ck(self.cu.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, None, params, None))
        self.ck(self.cu.cuCtxSynchronize())


def le_words(v): return v.to_bytes(32, "little")


def unit_test(cu, mod):
    import random
    rnd = random.Random(7)
    edge = [0, 1, 2, P - 1, P - 2, 2**255, 2**256 - 2**32 - 978, (1 << 224) - 1, 0x1000003D0, 2**32 - 1,
            int("FFFFFFFF" * 7 + "00000000", 16) % P, int("00000001" + "FFFFFFFF" * 7, 16)]
    vals = []
    for x in edge:
        for y in edge: vals.append((x, y))
    while len(vals) < 1 << 18:
        r = rnd.random()
        x = rnd.randrange(P) if r < .7 else P - 1 - rnd.randrange(2**rnd.randrange(1, 64))
        y = rnd.randrange(P) if rnd.random() < .7 else rnd.choice(edge)
        vals.append((x, y))
    n = len(vals)
    buf = b"".join(le_words(x) + le_words(y) for x, y in vals)
    din, dout = cu.alloc(len(buf)), cu.alloc(len(buf))
    cu.h2d(din, buf)
    fn = cu.func(mod, "unit")
    cu.launch(fn, (n + 255) // 256, 256, [c_uint64(din), c_uint64(dout), ctypes.c_uint32(n)])
    out = cu.d2h(dout, len(buf))
    bad_mul = bad_sqr = 0
    for k, (x, y) in enumerate(vals):
        m = int.from_bytes(out[64 * k:64 * k + 32], "little")
        q = int.from_bytes(out[64 * k + 32:64 * k + 64], "little")
        bad_mul += m != x * y % P
        bad_sqr += q != x * x % P
    print(f"UNIT {'PASS' if not bad_mul and not bad_sqr else 'FAIL'}: {n} cases, "
          f"mul mismatches {bad_mul}, sqr mismatches {bad_sqr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--out", default="found", help="directory for found keys (default: ./found)")
    ap.add_argument("--batch", type=int, default=511)
    ap.add_argument("--tpb", type=int, default=256)
    ap.add_argument("--blocks-per-sm", type=int, default=16)
    ap.add_argument("--secs", type=float, default=2.0, help="target seconds per launch")
    ap.add_argument("--test", action="store_true", help="verify every hit for a short run, then exit")
    ap.add_argument("--max-found", type=int, default=1)
    ap.add_argument("--bench", type=float, default=0, help="run N seconds and report throughput")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cfield", action="store_true", help="use the plain-C field arithmetic")
    ap.add_argument("--minb", type=int, default=2, help="__launch_bounds__ min blocks per SM")
    ap.add_argument("--orbit", type=int, default=191,
                    help="orbit kernel table size; 0 selects the plain batch kernel")
    ap.add_argument("--unit", action="store_true", help="test fe_mul/fe_sqr against Python, then exit")
    ap.add_argument("--audit", action="store_true", help="exact endomorphism checks; count filter misses")
    a = ap.parse_args()

    assert bech32_encode("a", b"") == "a12uel5l"  # BIP-173 test vector
    assert ec_mul(LAMBDA)[0] == BETA * G[0] % P, "lambda/beta mismatch"
    bad = [c for c in a.prefix if c not in CHARSET]
    if bad: sys.exit(f"characters not in bech32 alphabet: {bad}")
    nbits = 5 * len(a.prefix)
    assert nbits <= 64
    val = 0
    for c in a.prefix: val = val << 5 | CHARSET.index(c)
    mask = ((1 << nbits) - 1) << (64 - nbits)
    target = val << (64 - nbits)

    # The orbit kernel doesn't use the plain kernel's table; keep it tiny so both fit
    # in constant memory.
    B = 1 if a.orbit else a.batch
    cu = Cuda()
    sms = cu.attr(16)
    cc = (cu.attr(75), cu.attr(76))
    maxout = 1 << 16
    t0 = time.time()
    src = KERNEL.replace("@@FIELD@@", C_FIELD if a.cfield else ptxgen.field())
    mod = cu.compile(src, [f"-arch=sm_{cc[0]}{cc[1]}", f"-DBATCH={B}", f"-DTPB={a.tpb}", f"-DMINB={a.minb}"] + ([f"-DORBIT={a.orbit}"] if a.orbit else []) + [
                              f"-DMAXOUT={maxout}"] + (["-DAUDIT"] if a.audit else []) + (["--ptxas-options=-v"] if a.verbose else []), a.verbose)
    fn = cu.func(mod, "search_orbit" if a.orbit else "search")
    print(f"compiled for sm_{cc[0]}{cc[1]} in {time.time() - t0:.1f}s; {sms} SMs", flush=True)

    if a.unit:
        unit_test(cu, mod)
        return

    tbl = [ec_mul(1)]
    for _ in range(B - 1): tbl.append(ec_add(tbl[-1], G))
    tbl.append(ec_mul(2 * B + 1))
    cu.set_global(mod, "TGX", b"".join(le_words(p[0]) for p in tbl))
    cu.set_global(mod, "TGY", b"".join(le_words(p[1]) for p in tbl))
    cu.set_global(mod, "CBETA", le_words(BETA))
    cu.set_global(mod, "ZERO8", le_words(0))
    step = 2 * B + 1
    if a.orbit:
        rows = []
        q = None
        for k in range(a.orbit):
            q = ec_add(q, G)
            x, y = q
            b1, b2 = BETA, BETA * BETA % P
            rows += [x, b1 * x % P, b2 * x % P, x * x % P, b2 * x * x % P, b1 * x * x % P, pow(x, 3, P), y]
        cu.set_global(mod, "OT", b"".join(le_words(v) for v in rows))
        jp = ec_mul(2 * a.orbit + 1)
        cu.set_global(mod, "OJ", le_words(jp[0]) + le_words(jp[1]))
        step = 2 * a.orbit + 1
    cu.set_global(mod, "CBETA2", le_words(BETA * BETA % P))

    blocks = sms * a.blocks_per_sm
    T = blocks * a.tpb
    stride = 1 << 96  # spacing between threads' starting keys; no thread gets near it
    k0 = secrets.randbelow(N - (T + 1) * stride) + 1
    base = [k0 + t * stride for t in range(T)]
    D = ec_mul(stride)
    pts = [ec_mul(k0)]
    for _ in range(T - 1): pts.append(ec_add(pts[-1], D))
    xs = [le_words(p[0]) for p in pts]
    ys = [le_words(p[1]) for p in pts]
    soa = lambda ws: b"".join(w[4 * k:4 * k + 4] for k in range(8) for w in ws)
    dcx, dcy = cu.alloc(32 * T), cu.alloc(32 * T)
    cu.h2d(dcx, soa(xs)); cu.h2d(dcy, soa(ys))
    dout = cu.alloc(4 * (1 + 4 * maxout))
    print(f"{T} threads initialised in {time.time() - t0:.1f}s", flush=True)

    per_round = 3 * (6 * a.orbit + 1 if a.orbit else 2 * B + 1) * T
    expected = 32 ** len(a.prefix)
    os.makedirs(a.out, mode=0o700, exist_ok=True)
    done_rounds, checked, found, verified, failed, filter_miss, bound_viol = 0, 0, 0, 0, 0, 0, 0
    rounds = 1
    start = last = time.time()
    while True:
        cu.ck(cu.cu.cuMemsetD32_v2(dout, 0, 1))
        ts = time.time()
        cu.launch(fn, blocks, a.tpb, [c_uint64(dcx), c_uint64(dcy), ctypes.c_uint32(T),
                                       ctypes.c_uint32(rounds), c_uint64(mask), c_uint64(target),
                                       c_uint64(dout)])
        dt = time.time() - ts
        cnt = int.from_bytes(cu.d2h(dout, 4), "little")
        if cnt:
            raw = cu.d2h(dout, 4 * (1 + 4 * min(cnt, maxout)))
            for i in range(min(cnt, maxout)):
                tid, rnd, j, e = (int.from_bytes(raw[4 + 16 * i + 4 * q: 8 + 16 * i + 4 * q], "little")
                                  for q in range(4))
                if j == 0x40000001:
                    bound_viol += 1
                    continue
                if j == 0x40000000:
                    filter_miss += 1
                    continue
                if j >= 1 << 31: j -= 1 << 32
                orb, e = divmod(e, 3)
                sgn = -1 if j < 0 else 1
                k = (base[tid] + (done_rounds + rnd) * step + sgn * pow(LAMBDA, orb, N) * abs(j)) \
                    * pow(LAMBDA, e, N) % N
                npub = npub_of(k)
                if not npub.startswith("npub1" + a.prefix):
                    failed += 1
                    print(f"VERIFY FAILED tid={tid} rnd={rnd} j={j} e={e} got {npub}", flush=True)
                    continue
                verified += 1
                if not a.test:
                    found += 1
                    path = os.path.join(a.out, npub + ".txt")
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, "w") as f:
                        f.write(f"npub: {npub}\nnsec: {bech32_encode('nsec', k.to_bytes(32, 'big'))}\n"
                                f"hex_sk: {k:064x}\n")
                    print(f"FOUND {npub} -> {path}", flush=True)
            if cnt > maxout: print(f"warning: {cnt - maxout} hits dropped (buffer full)", flush=True)
        done_rounds += rounds
        checked += per_round * rounds
        now = time.time()
        rate = checked / (now - start)
        if a.test or (now - last > 30 and not a.bench) or found:
            print(f"{checked:.3e} keys  {rate / 1e6:,.0f} Mkeys/s  launch {dt:.2f}s x{rounds}  "
                  f"elapsed {now - start:,.0f}s  expected total {expected / rate / 60:,.1f} min  "
                  f"verified={verified} failed={failed} filter_miss={filter_miss} bound_viol={bound_viol}", flush=True)
            last = now
        if a.test and checked > 20 * expected: break
        if a.bench and now - start > a.bench:
            print(f"BENCH {rate / 1e6:.0f} Mkeys/s ({rate / 3e6:.0f} M points/s) B={B} tpb={a.tpb} "
                  f"bps={a.blocks_per_sm} minb={a.minb} verified={verified} failed={failed} filter_miss={filter_miss} "
                  f"bound_viol={bound_viol}", flush=True)
            break
        if found >= a.max_found: break
        if dt < a.secs / 2: rounds = max(1, int(rounds * min(4, a.secs / max(dt, 1e-3))))
    if a.test:
        exp_hits = checked / expected
        print(f"TEST {'PASS' if failed == 0 and verified > 0.5 * exp_hits else 'FAIL'}: "
              f"{verified} verified hits (expected about {exp_hits:.0f}), {failed} failures")


if __name__ == "__main__":
    main()
