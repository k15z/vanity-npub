# vanity-npub

Finds nostr npubs that start with a string you pick, using an NVIDIA GPU.

On an RTX 4060 Ti it checks 17.8 billion npubs per second. That's 3.8 times
faster than ngrind, the fastest open-source generator I could find, measured on
the same card. An 8-character prefix like `npub1nakam0t0` takes about a minute.

## Usage

You need Linux, an NVIDIA GPU with a recent driver, and Python 3.8 or newer. The
CUDA toolkit isn't required: the kernel is compiled at startup with NVRTC, which
you can install as a wheel next to the script.

```
pip install --target deps nvidia-cuda-nvrtc-cu12
python3 vanity_npub.py nakam0t0
```

If the CUDA toolkit is installed, the system `libnvrtc.so` is used instead.

The prefix is the part after `npub1`. It can only contain bech32 characters
(`qpzry9x8gf2tvdw0s3jn54khce6mua7l`), so `1`, `b`, `i` and `o` are out. Every
extra character makes the search 32 times longer.

When it finds a match it prints the npub and writes the npub, nsec and hex
private key to `found/<npub>.txt`, readable only by you. The key never touches
the terminal. Import it into your signer, then delete the file.

## Benchmarks

All numbers are npubs checked per second on one machine: RTX 4060 Ti 16 GB,
driver 580, Ubuntu 24.04, 28 CPU cores. Each GPU tool searched for a prefix it
couldn't realistically find, for 60 to 90 seconds.

| Tool | Hardware | npubs/s | vanity-npub is |
|---|---|---|---|
| vanity-npub | GPU | 17.8 G | |
| [ngrind](https://github.com/v0l/ngrind) `89f518a` | GPU | 4.66 G | 3.8x faster |
| [mocnpub](https://github.com/high-moctane/mocnpub) `e11884e` | GPU | 2.79 G | 6.4x faster |
| [rana](https://github.com/grunch/rana) `bb4d6b8` | CPU, 28 threads | 1.12 M | 16,000x faster |

Commands used:

```
python3 vanity_npub.py zzzzzzzzzzzz --bench 60
ngrind vanity --prefix qqqqqqqqqqqq --device cuda          # built with NGRIND_CUDA_ARCHS=89
mocnpub-main mine --prefix qqqqqqqqqqqq                    # rate from its progress line
rana -n nakam -c 28                                        # rate it reports after a match
```

ngrind and mocnpub were built with CUDA 12.9.

## How it works

Each GPU thread starts from a random private key and walks forward through
consecutive keys, adding a fixed point each step. An affine point addition
needs a modular inverse, so inverses are batched with Montgomery's trick: one
real inversion per batch and three multiplications per element. The points
C + Q and C - Q share a denominator, so each element yields two points. ngrind
does the same up to here.

secp256k1 has an endomorphism: for fixed constants beta and lambda, the point
for key lambda*k is (beta*x, y). An npub encodes only x, so each point gives
three npubs: x, beta*x and beta^2*x. Computing beta*x in full would cost a
multiplication per npub. Instead, the kernel estimates its top 32 bits from the
product columns that feed them (about 25 32-bit multiplies instead of 64) and
does the full multiplication only for the rare candidate that might match.
Because beta^2*x = -x - beta*x, the second estimate is a couple of additions.

The default kernel also groups table points. For a table point (a, b), the
points with x-coordinates a, beta*a and beta^2*a all share the same y. Their
three denominators against the current point multiply to a^3 - c^3, so a
single batch-inversion element covers six points and the slope numerators are
shared.

Field elements are eight 32-bit limbs. `ptxgen.py` generates multiplication and
squaring as PTX carry chains with even and odd partial products in separate
chains, which the compiler turns into fused multiply-add-with-carry
instructions.

The result costs about one 256-bit multiplication per npub. On this card the
integer multiply pipe is the bottleneck, so I'd expect careful work to get
another 20-30% at most.

## Checking it

```
python3 vanity_npub.py nak --test      # thousands of hits, each recomputed on the CPU
python3 vanity_npub.py q --unit        # field mul/sqr vs Python ints, 262,144 cases
python3 vanity_npub.py nakam0t0 --audit --bench 60
```

`--audit` runs the full beta multiplications alongside the estimates and counts
any estimate outside its error bound. Over 5 x 10^11 keys it found none.

Every match is recomputed on the CPU before it's saved, using this repo's own
secp256k1 code. If you plan to use a key for real, check it with another nostr
tool first.

## Options

`--orbit N` sets the grouped kernel's table size (default 191). `--orbit 0`
switches to the plain batch kernel, sized by `--batch`. `--tpb`,
`--blocks-per-sm` and `--minb` control the launch shape. `--cfield` swaps in
plain C field arithmetic for comparison.

## License

MIT
