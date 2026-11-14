# Downloads every wheel listed in a `pip install --dry-run --report` JSON into a folder, one file at a
# time, writing in 8 MiB chunks. Large chunks matter on this exFAT USB disk: 10 KiB appends (pip's
# own chunk size) run at ~1.9 MB/s, 4 MiB appends at ~11 MB/s, because exFAT allocates per append.
import json, os, sys, urllib.request
rep, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
CH = 8 << 20  # 8 MiB
for i in json.load(open(rep))["install"]:
    url = i["download_info"]["url"]; name = url.split("/")[-1].split("#")[0]
    dst = os.path.join(out, name)
    if os.path.exists(dst): continue
    with urllib.request.urlopen(url, timeout=120) as r:
        with open(dst + ".part", "wb") as f:
            while True:
                buf = r.read(CH)          # read() returns up to CH bytes; loop until a full chunk
                while buf and len(buf) < CH:
                    more = r.read(CH - len(buf))
                    if not more: break
                    buf += more
                if not buf: break
                f.write(buf)
    os.replace(dst + ".part", dst); print("ok", name, flush=True)
print("FETCH_DONE")
