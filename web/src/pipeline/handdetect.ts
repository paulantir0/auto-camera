// Port of auto_camera/handdetect.py: Google's MediaPipe hand landmarker and
// multiclass skin segmenter on one 480x270 frame, with the settings the Python
// pipeline uses. The models and MediaPipe's WebAssembly engine are files of
// this site (`hands/`), version 0.10.21 as in Python; nothing is fetched from
// anywhere else. The CPU engine is asked for, as Python runs it.

import { FilesetResolver, HandLandmarker, ImageSegmenter } from '@mediapipe/tasks-vision';
import { MAX_HANDS, MIN_CONFIDENCE, PROXY_HEIGHT, PROXY_WIDTH, SKIN_LEVEL } from './constants.ts';

const W = PROXY_WIDTH, H = PROXY_HEIGHT;

/** What the models found in one frame, as handnet.Detections holds it. */
export interface FrameHands {
  skin: Uint8Array;          // W*H, 1 where body or face skin is more likely than SKIN_LEVEL
  landmarks: Float32Array;   // MAX_HANDS * 21 * 2: x, y as shares of width and height; NaN where no hand
  scores: Float32Array;      // MAX_HANDS: handedness score, 0 where no hand
}

export interface DetectorFiles {
  base: string;              // folder with vision_wasm_internal.{js,wasm} (and the nosimd pair)
  landmarker: Uint8Array;    // hand_landmarker.task
  segmenter: Uint8Array;     // selfie_multiclass_256x256.tflite
  delegate?: 'CPU' | 'GPU';  // MediaPipe's engine for the models; CPU unless asked
}

export class HandDetector {
  private readonly landmarker: HandLandmarker;
  private readonly segmenter: ImageSegmenter;

  private constructor(landmarker: HandLandmarker, segmenter: ImageSegmenter) {
    this.landmarker = landmarker;
    this.segmenter = segmenter;
  }

  static async create(files: DetectorFiles): Promise<HandDetector> {
    const fileset = await FilesetResolver.forVisionTasks(files.base);
    const landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: files.landmarker, delegate: files.delegate ?? 'CPU' },
      runningMode: 'IMAGE',
      numHands: MAX_HANDS,
      minHandDetectionConfidence: MIN_CONFIDENCE,
      minHandPresenceConfidence: MIN_CONFIDENCE,
      minTrackingConfidence: MIN_CONFIDENCE,
    });
    const segmenter = await ImageSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: files.segmenter, delegate: files.delegate ?? 'CPU' },
      runningMode: 'IMAGE',
      outputConfidenceMasks: true,
      outputCategoryMask: false,
    });
    return new HandDetector(landmarker, segmenter);
  }

  /** The skin segmenter alone: 1 where skin is more likely than SKIN_LEVEL. */
  skin(image: ImageData, out = new Uint8Array(W * H)): Uint8Array {
    const result = this.segmenter.segment(image);
    try {
      const masks = result.confidenceMasks;
      if (!masks || masks.length < 4) throw new Error('the skin segmenter gave no confidence masks');
      if (masks[2].width !== W || masks[2].height !== H) throw new Error(`skin mask is ${masks[2].width}x${masks[2].height}, not ${W}x${H}`);
      const body = masks[2].getAsFloat32Array(), face = masks[3].getAsFloat32Array();
      for (let i = 0; i < W * H; i++) out[i] = Math.max(body[i], face[i]) > SKIN_LEVEL ? 1 : 0;
      return out;
    } finally {
      result.close();
    }
  }

  /** The hand landmarker alone. */
  hands(image: ImageData): { landmarks: Float32Array; scores: Float32Array } {
    const result = this.landmarker.detect(image);
    const landmarks = new Float32Array(MAX_HANDS * 21 * 2).fill(NaN);
    const scores = new Float32Array(MAX_HANDS);
    const count = Math.min(MAX_HANDS, result.landmarks.length, result.handedness.length);
    for (let h = 0; h < count; h++) {
      const points = result.landmarks[h];
      for (let p = 0; p < 21; p++) {
        landmarks[(h * 21 + p) * 2] = points[p].x;
        landmarks[(h * 21 + p) * 2 + 1] = points[p].y;
      }
      scores[h] = result.handedness[h][0].score;
    }
    return { landmarks, scores };
  }

  detect(image: ImageData): FrameHands {
    const skin = this.skin(image);
    return { skin, ...this.hands(image) };
  }

  close(): void {
    this.landmarker.close();
    this.segmenter.close();
  }
}

/** A BGR frame (as FFmpeg's bgr24 gives it) as the RGB image MediaPipe is given in Python (rgb24). */
export function bgrToImageData(bgr: Uint8Array, out?: ImageData): ImageData {
  const image = out ?? new ImageData(W, H);
  const d = image.data;
  for (let i = 0, o = 0; i < W * H; i++, o += 3) {
    d[4 * i] = bgr[o + 2];
    d[4 * i + 1] = bgr[o + 1];
    d[4 * i + 2] = bgr[o];
    d[4 * i + 3] = 255;
  }
  return image;
}
