#!/usr/bin/env bash
# Build the WebAssembly HEVC decoder that the page falls back to where the
# browser has none: FFmpeg's libavcodec with only its HEVC decoder (LGPL 2.1
# or later, no GPL parts), wrapped by wasm/hevcdec.c, compiled with
# Emscripten in its Docker image, twice: hevc.{js,wasm} single-threaded and
# hevc-mt.{js,wasm} with WebAssembly threads. The results are committed in
# wasm/ so that an ordinary build needs neither Docker nor Emscripten.
#
#   tools/build-hevc.sh            # about 5 minutes on 4 cores
#
# The container is limited to 4 CPUs and 3 GB of memory.
set -euo pipefail
FFMPEG=7.1.2
FFMPEG_SHA256=089bc60fb59d6aecc5d994ff530fd0dcb3ee39aa55867849a2bbc4e555f9c304
IMAGE=emscripten/emsdk:4.0.10
here=$(cd "$(dirname "$0")/.." && pwd)
work=${WORK:-$here/wasm/.build}
mkdir -p "$work"
tarball="$work/ffmpeg-$FFMPEG.tar.xz"
if [[ ! -f "$tarball" ]]; then
  curl -sSfL -o "$tarball" "https://ffmpeg.org/releases/ffmpeg-$FFMPEG.tar.xz"
fi
echo "$FFMPEG_SHA256  $tarball" | sha256sum -c -

docker run --rm --cpus 4 --memory 3g -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$work":/work -v "$here/wasm":/src "$IMAGE" bash -euo pipefail -c '
FFMPEG='"$FFMPEG"'
EXPORTS=_dec_open,_dec_send,_dec_receive,_dec_reset,_dec_picture,_dec_width,_dec_height,_dec_pts,_dec_matrix,_dec_full_range,_dec_close,_dec_alloc,_dec_free,_malloc,_free
for kind in st mt; do
  if [[ $kind == mt ]]; then flags="-O3 -msimd128 -pthread"; threads="--enable-pthreads"; else flags="-O3 -msimd128"; threads="--disable-pthreads"; fi
  dir=/work/$kind; prefix=/work/prefix-$kind
  if [[ ! -f $prefix/lib/libavcodec.a ]]; then
    rm -rf $dir && mkdir -p $dir && tar xf /work/ffmpeg-$FFMPEG.tar.xz -C $dir --strip-components=1
    cd $dir
    emconfigure ./configure --prefix=$prefix --target-os=none --arch=x86_32 --enable-cross-compile \
      --cc=emcc --cxx=em++ --ar=emar --ranlib=emranlib --nm=emnm \
      --disable-asm --disable-x86asm --disable-inline-asm --disable-runtime-cpudetect --disable-autodetect \
      --disable-programs --disable-doc --disable-debug --disable-stripping --disable-network \
      --disable-everything --disable-avdevice --disable-avformat --disable-avfilter --disable-swresample --disable-swscale \
      --enable-decoder=hevc $threads \
      --extra-cflags="$flags" --extra-ldflags="$flags" > /work/configure-$kind.log
    emmake make -j4 > /work/make-$kind.log 2>&1
    emmake make install >> /work/make-$kind.log 2>&1
  fi
  if [[ $kind == mt ]]; then
    out=hevc-mt; extra="-pthread -sPTHREAD_POOL_SIZE=Module.threads+1 -sPTHREAD_POOL_SIZE_STRICT=2 -sDEFAULT_PTHREAD_STACK_SIZE=2MB"
  else
    out=hevc; extra=""
  fi
  emcc $flags /src/hevcdec.c -I$prefix/include $prefix/lib/libavcodec.a $prefix/lib/libavutil.a -o /work/$out.js \
    $extra -sMODULARIZE=1 -sEXPORT_NAME=createHevcDecoder -sENVIRONMENT=worker \
    -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB -sMAXIMUM_MEMORY=2GB -sSTACK_SIZE=2MB \
    -sDYNAMIC_EXECUTION=0 -sFILESYSTEM=0 -sINCOMING_MODULE_JS_API=threads,mainScriptUrlOrBlob,print,printErr,locateFile \
    -sEXPORTED_FUNCTIONS=$EXPORTS -sEXPORTED_RUNTIME_METHODS=HEAPU8,wasmMemory
  cp /work/$out.js /work/$out.wasm /src/
done
'
ls -la "$here"/wasm/hevc*.js "$here"/wasm/hevc*.wasm
