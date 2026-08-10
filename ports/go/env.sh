# Source this before building or running the Go port on Linux:
#
#   . ./env.sh
#
# The Windows counterpart (env.ps1) carries the same three settings; see its comments
# for why ORT_SO must point at the SAME ONNX Runtime version the reference uses.
# On Linux the OpenCV libraries come from the distro package, so only the runtime
# library path needs saying out loud.

: "${ORT_SO:=/usr/local/lib/libonnxruntime.so}"
export ORT_SO

# Provenance for the sidecar -- an env var, not a baked-in value, so the binary stays
# reproducible. Re-source after committing or the reported commit goes stale.
if [ -z "${RDOCS_COMMIT:-}" ]; then
    RDOCS_COMMIT=$(git -C "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." rev-parse --short HEAD 2>/dev/null) || RDOCS_COMMIT=
    export RDOCS_COMMIT
fi
echo "commit : ${RDOCS_COMMIT:-<unknown>}"

if [ ! -e "$ORT_SO" ]; then
    echo "warning: ORT_SO does not exist: $ORT_SO" >&2
fi

echo "ORT_SO : $ORT_SO"
