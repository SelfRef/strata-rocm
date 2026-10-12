# strata-rocm

[Strata](https://github.com/Niko1221/Strata) for AMD RDNA3 GPUs, as a container image:
`ghcr.io/selfref/strata-rocm`.

Strata is an inference engine for **Qwen3.8-Flash-Next** (125B MoE) and its fine-tunes on one consumer GPU.
It keeps the most-used experts in VRAM, computes the rest on the CPU from RAM, and reads the n-gram table
from disk. Upstream publishes an NVIDIA (CUDA) image only. This repository builds the **HIP engine** for
gfx1100 / gfx1101 / gfx1102 (RX 7900 / 7800 / 7700 / 7600), the way Strata's own `setup.py` builds it on a
Linux AMD machine. The container needs no compiler and no ROCm install on the host.

**Meant to run under [llama-swap](https://github.com/mostlygeek/llama-swap)**, as a sibling container that
llama-swap starts on the first request and stops on unload. It pairs with
[llama-swap-rdna](https://github.com/SelfRef/llama-swap-rdna) (`ghcr.io/selfref/llama-swap-rdna`), a llama-swap
image with llama.cpp builds for the same cards. It works just as well on its own with `docker run`.

## Tags

| tag | what |
|---|---|
| `latest` | the newest Strata release |
| `<version>` | e.g. `0.1.43`: that release, moves if the image definition changes |
| `<version>-<sha7>` | immutable: that release at that upstream commit |

CI checks Strata's latest release every 4 hours and builds only when it changed. A manual run can build
any upstream tag, branch or commit.

## Host requirements

- The kernel's `amdgpu` driver. ROCm comes inside the image (AMD's TheRock wheels, the version Strata pins).
- **The whole card.** Strata fills the VRAM with its expert cache; do not share the card with another model.
- **RAM:** 35–62 GB of the host's memory is locked (`--ulimit memlock=-1`), depending on the model size.
- **The model's GGUF files**, e.g. ISTA-DASLab's GSQ-RCO quantizations
  (`hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include "Q2_0/*"`).
  They can be mounted read-only.
- A CPU with AVX2.

Pass **only the card Strata should use**, with its original device names. ROCm finds a card by the
render-node number in the KFD topology, so a renamed node (`renderD129` mapped to `renderD128`) is not
found:

```sh
ls -l /dev/dri/by-path/          # pci-0000:0a:00.0-render -> ../renderD129, -card -> ../card2
--device /dev/kfd --device /dev/dri/renderD129 --device /dev/dri/card2
```

## Usage

### 1. Prepare a model (once)

Runs Strata's `setup.py` against GGUF files you already have: it writes the pack, downloads the MTP draft
layer (~5 GB, from the original Qwen checkpoint), and saves the server config to `/data`. All `setup.py`
options pass through (`--model`, `--family`, `--kv`, `--context`, `--vision cpu`, ...). The model files must
be mounted at the **same path** later for `serve`, because the config records it.

```sh
docker run --rm --init \
  --device /dev/kfd --device /dev/dri/renderD129 --device /dev/dri/card2 \
  --ipc=host --ulimit memlock=-1 \
  -v strata-data:/data \
  -v /path/to/models:/models:ro \
  ghcr.io/selfref/strata-rocm:latest \
  prepare --family qwen --model Q2_0 --gguf-dir /models/Q2_0 \
          --vision no --kv int8 --context 40960
```

### 2. Serve

```sh
docker run --rm --init \
  --device /dev/kfd --device /dev/dri/renderD129 --device /dev/dri/card2 \
  --ipc=host --ulimit memlock=-1 \
  -v strata-data:/data -v /path/to/models:/models:ro \
  -p 8090:8090 \
  ghcr.io/selfref/strata-rocm:latest \
  serve q2_0 --port 8090
```

OpenAI API at `http://<host>:8090/v1` (`/v1/chat/completions`, `/v1/models`), and Anthropic's at `/v1/messages`.
Add `--api-key <key>` before exposing it to a network.

### 3. Under llama-swap

llama-swap starts the container per model and stops it on unload. Joining llama-swap's network namespace
makes `${PORT}` reachable on localhost without publishing anything. This needs the host's docker socket and
CLI inside the llama-swap container (`/var/run/docker.sock` and the `docker` binary mounted), which makes it
root-equivalent on the host.

```yaml
models:
  qwen38-flash-strata:
    name: "Qwen3.8 Flash-Next · Strata"
    proxy: "http://127.0.0.1:${PORT}"
    cmd: >
      docker run --name ${MODEL_ID} --init --rm
        --network container:<llama-swap container name>
        --device /dev/kfd --device /dev/dri/renderD129 --device /dev/dri/card2
        --ipc=host --ulimit memlock=-1:-1 --oom-score-adj 1000
        -e STRATA_ALLOWED_HOSTS=*
        -v strata-data:/data -v /path/to/models:/models:ro
        ghcr.io/selfref/strata-rocm:latest
        serve q2_0 --port ${PORT}
    cmdStop: docker stop -t 60 ${MODEL_ID}
    ttl: 1200
```

`STRATA_ALLOWED_HOSTS=*` is needed behind llama-swap: Strata's DNS-rebinding guard only answers to
localhost and IP addresses, and llama-swap forwards each client's `Host` header (e.g. `llama-swap:8080`),
so every request from another container would get a 403. It is safe here because the server is only
reachable through llama-swap's network namespace, behind llama-swap's own authentication.

Strata takes the whole card, so put it in a llama-swap group that is exclusive with the other models on that
GPU. The first start after a reboot loads ~38 GB of experts into RAM (~20 s from NVMe).

## Measured

Qwen3.8-Flash-Next GSQ-RCO **Q2_0**, RX 7900 XTX (24 GB), Ryzen 9 5950X (AVX2), DDR4-3334, Strata 0.1.43,
`--kv int8`, MTP draft depth 4, greedy, thinking off. llama.cpp's MoE expert cache on the same files and card
for comparison:

| | decode t/s | 32k-token prefill t/s |
|---|---|---|
| Strata (18 GiB expert cache, 95 % hit rate) | 97–112 | 1323 |
| llama.cpp Vulkan, `--cpu-moe --moe-cache-mib` + MTP head, `-ub 2048` | 49–56 | 482–502 |

## How the image is built

- Strata is checked out at the release tag, and its pinned Python packages go into `/opt/strata/.venv`.
- The HIP engine is compiled by calling `setup.py`'s own `build_engine_hip()`. It installs the TheRock ROCm
  wheels Strata pins for the gfx110X family, compiles with Strata's cmake flags, and stamps
  `engine/BUILD.json`, so `prepare` finds the engine already built.
- ggml's CPU code is built for a portable AVX2/FMA/F16C baseline (`STRATA_PORTABLE`) instead of the build
  machine's `-march=native`, which would bake in AVX-512 on many CI runners. Strata's own AVX-512 kernels are
  compiled per file and chosen at runtime either way.

Not included: other GPU families (RDNA4, Strix Halo, RDNA2). The ROCm wheels hold one family's libraries, so
each family would be its own image, with `HIP_ARCHS` and the TheRock index changed.

## License

The files in this repository are AGPL-3.0 ([LICENSE](LICENSE)). Strata is MIT (its repository lists the
few parts under other licenses). The models carry their own licenses.
