// A minimal HEVC decoder for browsers that have none: FFmpeg's libavcodec
// HEVC decoder (LGPL), nothing else, behind a handful of functions the page
// calls through WebAssembly. Built by web/tools/build-hevc.sh.
//
// Packets go in as they are stored in MP4/QuickTime (length-prefixed NAL
// units, with the hvcC box as extradata); pictures come out in display order
// as 8-bit I420 in one tightly packed buffer (10-bit video is shifted down).

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten/emscripten.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>

typedef struct {
  AVCodecContext *ctx;
  AVPacket *pkt;
  AVFrame *frame;
  uint8_t *out;      // the last picture, I420, width x height
  size_t out_size;
  int width, height; // of the last picture
  double pts;        // its timestamp, as given with its packet
  int matrix, full_range;
} Dec;

EMSCRIPTEN_KEEPALIVE Dec *dec_open(const uint8_t *extra, int extra_size, int threads) {
  const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
  if (!codec) return NULL;
  Dec *d = calloc(1, sizeof(Dec));
  if (!d) return NULL;
  d->ctx = avcodec_alloc_context3(codec);
  d->pkt = av_packet_alloc();
  d->frame = av_frame_alloc();
  if (!d->ctx || !d->pkt || !d->frame) goto fail;
  if (extra_size > 0) {
    d->ctx->extradata = av_mallocz(extra_size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!d->ctx->extradata) goto fail;
    memcpy(d->ctx->extradata, extra, extra_size);
    d->ctx->extradata_size = extra_size;
  }
  d->ctx->thread_count = threads > 0 ? threads : 1;
  d->ctx->thread_type = threads > 1 ? FF_THREAD_FRAME : 0;
  d->ctx->pkt_timebase = (AVRational){1, 1000000};
  if (avcodec_open2(d->ctx, codec, NULL) < 0) goto fail;
  return d;
fail:
  av_frame_free(&d->frame);
  av_packet_free(&d->pkt);
  avcodec_free_context(&d->ctx);
  free(d);
  return NULL;
}

// 0: accepted; 1: not accepted, take pictures out first and send it again; < 0: error.
EMSCRIPTEN_KEEPALIVE int dec_send(Dec *d, const uint8_t *data, int size, double pts, int key) {
  int r;
  if (!data) {
    r = avcodec_send_packet(d->ctx, NULL);
    return r == AVERROR_EOF ? 0 : r;
  }
  if (av_new_packet(d->pkt, size) < 0) return -12;
  memcpy(d->pkt->data, data, size);
  d->pkt->pts = (int64_t)pts;
  d->pkt->dts = AV_NOPTS_VALUE;
  if (key) d->pkt->flags |= AV_PKT_FLAG_KEY;
  r = avcodec_send_packet(d->ctx, d->pkt);
  av_packet_unref(d->pkt);
  if (r == AVERROR(EAGAIN)) return 1;
  return r;
}

static int to_i420(Dec *d) {
  AVFrame *f = d->frame;
  const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(f->format);
  if (!desc || desc->nb_components < 3 || desc->log2_chroma_w != 1 || desc->log2_chroma_h != 1 || (desc->flags & AV_PIX_FMT_FLAG_RGB)) return -2;
  int w = f->width, h = f->height, cw = (w + 1) / 2, ch = (h + 1) / 2;
  size_t need = (size_t)w * h + 2 * (size_t)cw * ch;
  if (need > d->out_size) {
    free(d->out);
    d->out = malloc(need);
    if (!d->out) { d->out_size = 0; return -12; }
    d->out_size = need;
  }
  int depth = desc->comp[0].depth;
  uint8_t *dst[3] = { d->out, d->out + (size_t)w * h, d->out + (size_t)w * h + (size_t)cw * ch };
  int dw[3] = { w, cw, cw }, dh[3] = { h, ch, ch };
  for (int p = 0; p < 3; p++) {
    for (int y = 0; y < dh[p]; y++) {
      const uint8_t *row = f->data[p] + (size_t)y * f->linesize[p];
      uint8_t *out = dst[p] + (size_t)y * dw[p];
      if (depth == 8) memcpy(out, row, dw[p]);
      else {
        const uint16_t *r16 = (const uint16_t *)row;
        int shift = depth - 8;
        for (int x = 0; x < dw[p]; x++) out[x] = (uint8_t)(r16[x] >> shift);
      }
    }
  }
  d->width = w;
  d->height = h;
  d->pts = (double)(f->pts != AV_NOPTS_VALUE ? f->pts : f->best_effort_timestamp);
  d->matrix = f->colorspace;
  d->full_range = f->color_range == AVCOL_RANGE_JPEG || f->format == AV_PIX_FMT_YUVJ420P;
  return 0;
}

// 0: a picture is ready (dec_picture etc.); 1: none yet, send more; 2: no more after a flush; < 0: error.
EMSCRIPTEN_KEEPALIVE int dec_receive(Dec *d) {
  int r = avcodec_receive_frame(d->ctx, d->frame);
  if (r == AVERROR(EAGAIN)) return 1;
  if (r == AVERROR_EOF) return 2;
  if (r < 0) return r;
  r = to_i420(d);
  av_frame_unref(d->frame);
  return r;
}

// Forget everything sent so far, to start again at another keyframe.
EMSCRIPTEN_KEEPALIVE void dec_reset(Dec *d) { avcodec_flush_buffers(d->ctx); }

EMSCRIPTEN_KEEPALIVE uint8_t *dec_picture(Dec *d) { return d->out; }
EMSCRIPTEN_KEEPALIVE int dec_width(Dec *d) { return d->width; }
EMSCRIPTEN_KEEPALIVE int dec_height(Dec *d) { return d->height; }
EMSCRIPTEN_KEEPALIVE double dec_pts(Dec *d) { return d->pts; }
EMSCRIPTEN_KEEPALIVE int dec_matrix(Dec *d) { return d->matrix; }
EMSCRIPTEN_KEEPALIVE int dec_full_range(Dec *d) { return d->full_range; }

EMSCRIPTEN_KEEPALIVE void dec_close(Dec *d) {
  if (!d) return;
  avcodec_free_context(&d->ctx);
  av_frame_free(&d->frame);
  av_packet_free(&d->pkt);
  free(d->out);
  free(d);
}

EMSCRIPTEN_KEEPALIVE uint8_t *dec_alloc(int size) { return malloc(size); }
EMSCRIPTEN_KEEPALIVE void dec_free(uint8_t *p) { free(p); }
