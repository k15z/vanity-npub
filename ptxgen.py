"""Generate the secp256k1 field multiply, square and reduce as PTX carry chains.

Each carry chain is emitted as a single asm block, so the carry flag never has
to survive between separate asm statements.
"""


def chain(ops, tail=(), dest="t"):
    """A carry chain of mad.{lo,hi} ops into dest[pos], then addc into each tail position.

    ops = [("lo"|"hi", pos, x, y)]: dest[pos] += lo/hi(x*y) + carry.
    """
    outs, ins, body = [], [], []
    for k, (kind, pos, x, y) in enumerate(ops):
        body.append((f"{'madc' if k else 'mad'}.{kind}.cc.u32", pos, x, y))
        for e in (x, y):
            if e not in ins: ins.append(e)
    for k, pos in enumerate(tail):
        body.append(("addc.cc.u32" if k < len(tail) - 1 else "addc.u32", pos, None, None))
    for _, pos, _, _ in body:
        if pos not in outs: outs.append(pos)
    n = len(outs)
    lines = []
    for op, pos, x, y in body:
        d = f"%{outs.index(pos)}"
        if x is None:
            lines.append(f"{op} {d}, {d}, 0;")
        else:
            lines.append(f"{op} {d}, %{n + ins.index(x)}, %{n + ins.index(y)}, {d};")
    return ('  asm("' + "\\n\\t".join(lines) + '"\n      : ' +
            ", ".join(f'"+r"({dest}[{p}])' for p in outs) + "\n      : " +
            ", ".join(f'"r"({e})' for e in ins) + ");\n")


def mul512():
    """t[0..15] = a*b. Per row b[i]: one chain over even j, one over odd j (Emmart-Weems).

    Invariant before row i: t[i+8] <= 1 and t[k] == 0 for k > i+8, so each chain's
    final addc never overflows.
    """
    c = ["DEV void mul512(u32 t[18], const u32 a[8], const u32 b[8]) {\n",
         "#pragma unroll\n  for (int k = 0; k < 18; k++) t[k] = 0;\n"]
    for i in range(8):
        for js in ((0, 2, 4, 6), (1, 3, 5, 7)):
            ops = []
            for j in js:
                ops += [("lo", i + j, f"a[{j}]", f"b[{i}]"), ("hi", i + j + 1, f"a[{j}]", f"b[{i}]")]
            c.append(chain(ops, (ops[-1][1] + 1,)))
    c.append("}\n")
    return "".join(c)


def sqr512():
    """t[0..15] = a^2: cross products a[i]a[j] (i<j), doubled, plus the diagonal.

    Row i is split by the parity of (j - i). The chain that ends lower runs first,
    which keeps the mul512 invariant: words past the touched region are 0 or 1.
    """
    c = ["DEV void sqr512(u32 t[18], const u32 a[8]) {\n",
         "#pragma unroll\n  for (int k = 0; k < 18; k++) t[k] = 0;\n"]
    for i in range(7):
        chains = []
        for par in (1, 2):
            ops = []
            for j in range(i + par, 8, 2):
                ops += [("lo", i + j, f"a[{i}]", f"a[{j}]"), ("hi", i + j + 1, f"a[{i}]", f"a[{j}]")]
            if ops: chains.append(ops)
        chains.sort(key=lambda ops: ops[-1][1])
        for ops in chains:
            c.append(chain(ops, (ops[-1][1] + 1,)))
    dbl = ["add.cc.u32 %0, %0, %0;"] + [f"addc.cc.u32 %{k}, %{k}, %{k};" for k in range(1, 15)] + \
          ["addc.u32 %15, %15, %15;"]
    c.append('  asm("' + "\\n\\t".join(dbl) + '"\n      : ' +
             ", ".join(f'"+r"(t[{k}])' for k in range(16)) + ");\n")
    diag = []
    for i in range(8):
        diag += [("lo", 2 * i, f"a[{i}]", f"a[{i}]"), ("hi", 2 * i + 1, f"a[{i}]", f"a[{i}]")]
    c.append(chain(diag))
    c.append("}\n")
    return "".join(c)


def reduce():
    """r = t mod p (canonical), p = 2^256 - 0x1000003D1, via hi*2^256 == hi*(2^32 + 977)."""
    c = ["DEV void fe_reduce(u32 r[8], const u32 t[18]) {\n  u32 s[10];\n  const u32 K = 977u;\n",
         "#pragma unroll\n  for (int k = 0; k < 8; k++) s[k] = t[k];\n  s[8] = 0; s[9] = 0;\n"]
    c.append(chain([("lo", k, f"t[{8 + k}]", "K") for k in range(8)], (8,), dest="s"))
    c.append(chain([("hi", k + 1, f"t[{8 + k}]", "K") for k in range(8)], (9,), dest="s"))
    lines = ["add.cc.u32 %0, %0, %9;"] + [f"addc.cc.u32 %{k}, %{k}, %{9 + k};" for k in range(1, 8)] + \
            ["addc.u32 %8, %8, 0;"]
    c.append('  asm("' + "\\n\\t".join(lines) + '"\n      : ' +
             ", ".join(f'"+r"(s[{k}])' for k in range(1, 10)) + "\n      : " +
             ", ".join(f'"r"(t[{8 + k}])' for k in range(8)) + ");\n")
    c.append(SECOND_FOLD)
    return "".join(c)


SECOND_FOLD = r'''  // second fold: v = s8 + s9*2^32 (< 2^34): r = s[0..7] + v*977 + (v << 32)
  u32 u = __umulhi(s[8], K) + s[9] * K, c1, c2;
#pragma unroll
  for (int k = 0; k < 8; k++) r[k] = s[k];
  asm("mad.lo.cc.u32 %0, %9, %10, %0;\n\taddc.cc.u32 %1, %1, %11;\n\taddc.cc.u32 %2, %2, 0;\n\t"
      "addc.cc.u32 %3, %3, 0;\n\taddc.cc.u32 %4, %4, 0;\n\taddc.cc.u32 %5, %5, 0;\n\t"
      "addc.cc.u32 %6, %6, 0;\n\taddc.cc.u32 %7, %7, 0;\n\taddc.u32 %8, 0, 0;"
      : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]), "+r"(r[4]), "+r"(r[5]), "+r"(r[6]), "+r"(r[7]),
        "=r"(c1)
      : "r"(s[8]), "r"(K), "r"(u));
  asm("add.cc.u32 %0, %0, %8;\n\taddc.cc.u32 %1, %1, %9;\n\taddc.cc.u32 %2, %2, 0;\n\t"
      "addc.cc.u32 %3, %3, 0;\n\taddc.cc.u32 %4, %4, 0;\n\taddc.cc.u32 %5, %5, 0;\n\t"
      "addc.cc.u32 %6, %6, 0;\n\taddc.u32 %7, 0, 0;"
      : "+r"(r[1]), "+r"(r[2]), "+r"(r[3]), "+r"(r[4]), "+r"(r[5]), "+r"(r[6]), "+r"(r[7]), "=r"(c2)
      : "r"(s[8]), "r"(s[9]));
  if (c1 | c2) fe_add_c(r);
  fe_norm(r);
}
'''


def field():
    return (mul512() + "\n" + sqr512() + "\n" + reduce() + "\n" +
            "DEV void fe_mul(u32 r[8], const u32 a[8], const u32 b[8]) { u32 t[18]; mul512(t, a, b); fe_reduce(r, t); }\n"
            "DEV void fe_sqr(u32 r[8], const u32 a[8]) { u32 t[18]; sqr512(t, a); fe_reduce(r, t); }\n")


if __name__ == "__main__":
    print(field())
