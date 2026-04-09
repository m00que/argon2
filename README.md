# argon2-kraken

GPU-accelerated Argon2 password cracking tool, supporting CUDA (NVIDIA) and OpenCL.

## Download

Grab the latest Windows x64 binary from [Releases](https://github.com/m00que/argon2/releases).  
No install needed — VC++ Redistributable is bundled.

---

## Modes

### Association Attack (default)

Pairs each line in `leftlist` with the corresponding line in `wordlist` and verifies the hash.  
Use this when you already have candidate (username, password) pairs.

```
argon2-kraken.exe <mode> [-b <batchSize>] <leftlist> <wordlist> <potfile>
```

### Dictionary Attack (`-d`)

Tries every password in a wordlist against a single hash.  
Classic brute-force / wordlist attack mode.

```
argon2-kraken.exe <mode> [-b <batchSize>] -d <hash> <wordlist> <potfile>
```

---

## Parameters

| Parameter | Description |
|-----------|-------------|
| `mode` | `cuda` (NVIDIA GPU) or `opencl` (any GPU) |
| `-b <n>` | GPU batch size — number of passwords pushed to GPU per round (default: **32**) |
| `-d <hash>` | Enable dictionary attack mode; specify the target hash inline |
| `leftlist` | File with hashes (association attack) |
| `wordlist` | File with candidate passwords, one per line |
| `potfile` | Output file — cracked results written here |

---

## Tuning `-b` (Batch Size)

Each batch requires `batchSize × m` KB of GPU memory, where `m` is the Argon2 memory parameter from the hash.

**Example:** hash with `m=65536` (64 MB per password)

| `-b` | VRAM required |
|------|--------------|
| 32 (default) | ~2 GB |
| 64 | ~4 GB |
| 115 | ~7.4 GB |

- The tool will **warn** if the requested batch exceeds 90% of your GPU's total VRAM.
- If you get an out-of-memory error, lower `-b`.
- Increasing `-b` improves throughput roughly linearly — use the highest value your GPU allows.

---

## Examples

Verify a known password:
```
argon2-kraken.exe cuda -d "$argon2id$v=19$m=65536,t=3,p=1$..." passwords.txt result.txt
```

Dictionary attack with larger batch (8 GB GPU):
```
argon2-kraken.exe cuda -b 100 -d "$argon2id$v=19$m=65536,t=3,p=1$..." rockyou.txt result.txt
```

---

## Build from Source

Requirements: CMake 3.18+, MSVC 2019+, CUDA Toolkit (optional), OpenCL SDK.

```
git clone --recursive https://github.com/m00que/argon2
cd argon2
cmake -B build-msvc -DCMAKE_BUILD_TYPE=Release
cmake --build build-msvc --config Release
```
