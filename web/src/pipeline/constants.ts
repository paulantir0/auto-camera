// Every threshold of the selection logic, under the names the Python modules
// use (auto_camera/*.py at commit 60a5811 of branch fast-selector). When a
// value changes there, change it here; test/video.test.ts compares this
// file with the Python source.

// --- proxy.py -------------------------------------------------------------
export const PROXY_WIDTH = 480;
export const PROXY_HEIGHT = 270;

// --- analysis.py ----------------------------------------------------------
export const THUMB_H = 54;
export const THUMB_W = 96;
export const CELL = 10;
export const GRID_H = PROXY_HEIGHT / CELL; // 27
export const GRID_W = PROXY_WIDTH / CELL; // 48
export const CELL_THRESHOLD = 6;
export const STRONG_THRESHOLD = 25;
export const DETAIL_SCALE = 1.0;
export const MEDIAN_SAMPLE_FRAMES = 240;
export const PAPER_CHROMA_TOLERANCE = 9;
export const PAPER_VALUE_SHARE = 0.7;
export const TEXT_LEVEL = 0.5;
export const DIP_LEVEL = 0.5;
export const DIP_SHARE = 0.3;
export const MIN_DIP_RATE = 0.02;
export const MIN_TEXT_FILL = 0.4;
export const ACTIVITY_SHARE = 0.35;
export const MARGIN_X = 0.09;
export const MARGIN_Y = 0.13;
export const ROW_SLACK = 0.1;
export const PRINT_LEVEL = 0.5;
export const MIN_PRINT_CELLS = 40;

// --- hands.py -------------------------------------------------------------
export const SAMPLE_FRAMES = 300;
export const BINS = 64;
export const LOW = 64;
export const SPAN = 128;
export const FOREGROUND = 35;
export const MIN_LIGHT = 70;
export const LIKELIHOOD_RATIO = 3.0;
export const MIN_BLOB = 60;
export const MIN_SAMPLES = 4000;
export const CHROMA_CHANGE = 10;
export const PRINT_PARTS = 1;
export const PRINT_FRAMES = 300;
export const MIN_PART_FRAMES = 30;
export const PRINT_CALM = 0.03;
export const HANDS_PRINT_LEVEL = 0.4; // hands.PRINT_LEVEL (analysis.PRINT_LEVEL is another)
export const PRINT_SHARE = 0.35;
export const PRINT_CHANGE = 0.4;
export const PRINT_REACH = 15;
export const MIN_SEEN = 10;
export const PRINT_INSIDE = 4;
export const PRINT_JOIN = 11;
export const PRINT_NEAR = 10;
export const PRINT_SHARED = 0.2;

// --- handnet.py and handdetect.py (the trained hand models) ----------------
export const HANDNET_MIN_SCORE = 0.5; // handnet.MIN_SCORE (ocr.MIN_SCORE is another)
export const MIN_SKIN = 12;
export const SKIN_OPEN = 5;
// the landmark silhouette, which Python draws only for detections without skin masks; not used here
export const FINGER_WIDTH = 0.24;
export const THUMB_WIDTH = 0.28;
export const PALM_MARGIN = 0.10;
export const SKIN_LEVEL = 0.5;
export const MAX_HANDS = 4;
export const MIN_CONFIDENCE = 0.3;
export const VERSION = 2; // handdetect.VERSION, of the detections Python caches

// --- spreads.py -----------------------------------------------------------
export const LAND_COVERAGE = 0.90;
export const LIFT_COVERAGE = 0.90;
export const SHARP_ENOUGH = 0.60;
export const HAND_STEP = 0.0025;
export const SAME_SPREAD_CHANGED = 0.03;
export const SAME_SPREAD_SHARE = 0.3;
export const FAR_APART = 8;
export const CALM_MOTION = 0.15;
export const CALM_SLACK = 0.10;
export const WEAK_LAND_COVERAGE = 0.75;
export const LONG_INTERVAL = 1.8;
export const HIDDEN_LEVEL = 0.4;
export const HIDDEN_DROP = 0.5;
export const HIDDEN_GROUP = 4;
export const HIDDEN_PAPER = 0.9;
export const CLEAR_MARGIN = 0.005;
export const HAND_NEAR = 2;
export const NEAR_MOTION = 0.10;
export const PAGE_HAND_STEP = 0.01;
export const UNSETTLED_STEPS = 8;
export const UNSETTLED = 0.15;
export const TWO_FRAMES_STEPS = 2; // spreads.TWO_FRAMES is False in Python: each spread is one frame, here too

// --- repeats.py -----------------------------------------------------------
export const MATCH_WIDTH = 1536;
export const LOOK_BACK = 3;
export const STATIC_EDGE = 60;
export const ELSEWHERE_INLIERS = 18;
export const ELSEWHERE_WINDOW = 8;
export const LEGIBLE_SHARE = 0.25;
export const LEGIBLE_USUAL = 0.5;
export const PAGE_PAPER = 0.3;
export const MAX_SCALE = 1.2;
export const MAX_ROTATION = 12.0;
export const PAGE_INLIERS = 12;
export const SPREAD_INLIERS = 60;
export const MIN_PAGE_KEYPOINTS = 60;
export const HALF_PAGE_INLIERS = 40;
export const UNSETTLED_MOTION = 0.15;
export const REPEATS_FAR_APART = 8; // repeats.FAR_APART
export const BASELINE_PAIRS = 120;
// ORB and matching settings written inline in repeats.py
export const ORB_FEATURES = 2500; // cv2.ORB_create(nfeatures=2500, ...)
export const ORB_FAST_THRESHOLD = 12; // ... fastThreshold=12
export const RATIO_TEST = 0.75; // pair[0].distance < 0.75 * pair[1].distance
export const MIN_GOOD_MATCHES = 12; // if len(good) < 12: return 0
export const MIN_KEYPOINTS = 20; // len(keys_a) < 20 or len(keys_b) < 20
export const RANSAC_REPROJ_THRESHOLD = 6.0;

// --- export.py ------------------------------------------------------------
export const EXPORT_UNSETTLED_MOTION = 0.15; // export.UNSETTLED_MOTION
export const HAND_ON_PRINT = 0.003;

// find_valleys defaults (analysis.py)
export const VALLEY_WINDOW = 3;
export const VALLEY_CEILING = 0.15;
export const VALLEY_DIP_CEILING = 0.45;
export const VALLEY_MIN_PROMINENCE = 0.20;
export const VALLEY_REACH = 22;

// --- pagenumbers.py (the printed-number check, optional in the browser) ---
export const SHIFT_COST = 0.6;
export const MIN_SUPPORT = 2;
export const JUMP_COST = 3.0;
export const RESTART_SPREADS = 5;
export const NUMBER_LOOK_BACK = 4;

// --- audit.py -------------------------------------------------------------
export const MIN_READ = 30;
export const MIN_READ_SHARE = 0.6;
export const STATIC_SHARE = 0.08;
export const STATIC_MIN = 6;
export const PLACE_WIDTH = 0.06;
export const LINE_END_GAIN = 1.15;

// --- ocr.py ---------------------------------------------------------------
export const MIN_SCORE = 0.6;
export const DET_MAX_SIDE = 1600;
export const MAX_ASPECT = 4.5;
export const END_ASPECT = 4.0;
export const MIN_HEIGHT = 8;
export const BAND_OUT = 0.12;
export const BAND_IN = 0.07;
