# Streams the Ollama release tarball (decompressed tar on stdin) onto exFAT, which has no symlinks.
# Pass "map" prints JSON {regular_file: [link names pointing to it]}; pass "extract" writes each regular
# file under its soname-style link names (e.g. libcublasLt.so.13) instead of creating symlinks.
# Skipped: cuda_v12 (driver 595 supports CUDA 13), vulkan, mlx, rocm (not used on an NVIDIA laptop GPU).
import sys, tarfile, json, os, re, shutil
mode, dest = sys.argv[1], sys.argv[2]
SKIP = re.compile(r"cuda_v12|vulkan|mlx|rocm")
if mode == "map":
    links = {}
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tf:
        for m in tf:
            if m.issym():
                tgt = os.path.normpath(os.path.join(os.path.dirname(m.name), m.linkname))
                links.setdefault(tgt, []).append(m.name)
    # Resolve link chains (a -> b -> real) so every name maps to the final regular file.
    flat = {}
    def resolve(n, seen=()):
        for real, names in links.items():
            if n in names: return resolve(real, seen + (n,))
        return n
    allnames = [n for v in links.values() for n in v]
    for n in allnames:
        flat.setdefault(resolve(n), []).append(n)
    json.dump(flat, open(dest, "w"))
else:
    flat = json.load(open(sys.argv[3]))
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or SKIP.search(m.name): continue
            names = flat.get(m.name, [])
            # Keep soname aliases (end in .so.<major> or .so.<major>.<minor>) that the loader asks for;
            # if the file has no aliases, write it under its own name.
            # Small non-CUDA libs (ggml/llama) also get their bare .so name, since Ollama may look them up by it.
            wanted = [n for n in names if re.search(r"\.so\.\d+(\.\d+)?$", n) or ("cuda_" not in n and m.size < 50_000_000)] or [m.name]
            src = tf.extractfile(m)
            first = os.path.join(dest, wanted[0]); os.makedirs(os.path.dirname(first), exist_ok=True)
            with open(first, "wb") as f: shutil.copyfileobj(src, f, 1 << 20)
            os.chmod(first, m.mode)
            for extra in wanted[1:]:
                shutil.copyfile(first, os.path.join(dest, extra)); os.chmod(os.path.join(dest, extra), m.mode)
            print("wrote", wanted, flush=True)
