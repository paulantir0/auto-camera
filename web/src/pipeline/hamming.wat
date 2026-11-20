;; Brute-force 2-nearest-neighbour matching of 32-byte binary descriptors by
;; Hamming distance, as cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a, b, k=2).
;; Ties keep the lower train index first, as OpenCV does.
;;
;; knn2(a, na, b, nb, out): for each query i writes three i32 at out + 12 i:
;; index of the nearest train descriptor, its distance, the second distance.
(module
  (memory (export "memory") 2)

  (func $dist (param $q0 v128) (param $q1 v128) (param $p i32) (result v128)
    ;; per-descriptor partial sums as 4 x i32
    (i32x4.extadd_pairwise_i16x8_u
      (i16x8.extadd_pairwise_i8x16_u
        (i8x16.add
          (i8x16.popcnt (v128.xor (local.get $q0) (v128.load (local.get $p))))
          (i8x16.popcnt (v128.xor (local.get $q1) (v128.load offset=16 (local.get $p))))))))

  (func (export "knn2") (param $a i32) (param $na i32) (param $b i32) (param $nb i32) (param $out i32)
    (local $i i32) (local $j i32) (local $p i32) (local $q0 v128) (local $q1 v128)
    (local $s0 v128) (local $s1 v128) (local $s2 v128) (local $s3 v128) (local $x v128) (local $y v128) (local $d v128)
    (local $best i32) (local $second i32) (local $bestj i32) (local $k i32) (local $dk i32) (local $nb4 i32)
    (local.set $nb4 (i32.and (local.get $nb) (i32.const -4)))
    (block $done
      (loop $queries
        (br_if $done (i32.ge_u (local.get $i) (local.get $na)))
        (local.set $q0 (v128.load (i32.add (local.get $a) (i32.shl (local.get $i) (i32.const 5)))))
        (local.set $q1 (v128.load offset=16 (i32.add (local.get $a) (i32.shl (local.get $i) (i32.const 5)))))
        (local.set $best (i32.const 1000))
        (local.set $second (i32.const 1000))
        (local.set $bestj (i32.const -1))
        (local.set $j (i32.const 0))
        (local.set $p (local.get $b))
        ;; four train descriptors at a time
        (block $end4
          (loop $train4
            (br_if $end4 (i32.ge_u (local.get $j) (local.get $nb4)))
            (local.set $s0 (call $dist (local.get $q0) (local.get $q1) (local.get $p)))
            (local.set $s1 (call $dist (local.get $q0) (local.get $q1) (i32.add (local.get $p) (i32.const 32))))
            (local.set $s2 (call $dist (local.get $q0) (local.get $q1) (i32.add (local.get $p) (i32.const 64))))
            (local.set $s3 (call $dist (local.get $q0) (local.get $q1) (i32.add (local.get $p) (i32.const 96))))
            ;; x = [s0a, s0b, s1a, s1b], y = [s2a, s2b, s3a, s3b]
            (local.set $x (i32x4.add
              (i8x16.shuffle 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27 (local.get $s0) (local.get $s1))
              (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31 (local.get $s0) (local.get $s1))))
            (local.set $y (i32x4.add
              (i8x16.shuffle 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27 (local.get $s2) (local.get $s3))
              (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31 (local.get $s2) (local.get $s3))))
            (local.set $d (i32x4.add
              (i8x16.shuffle 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27 (local.get $x) (local.get $y))
              (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31 (local.get $x) (local.get $y))))
            (if (v128.any_true (i32x4.lt_s (local.get $d) (i32x4.splat (local.get $second))))
              (then
                (local.set $k (i32.const 0))
                (loop $lanes
                  (local.set $dk
                    (if (result i32) (i32.eqz (local.get $k)) (then (i32x4.extract_lane 0 (local.get $d)))
                      (else (if (result i32) (i32.eq (local.get $k) (i32.const 1)) (then (i32x4.extract_lane 1 (local.get $d)))
                        (else (if (result i32) (i32.eq (local.get $k) (i32.const 2)) (then (i32x4.extract_lane 2 (local.get $d)))
                          (else (i32x4.extract_lane 3 (local.get $d)))))))))
                  (if (i32.lt_s (local.get $dk) (local.get $best))
                    (then
                      (local.set $second (local.get $best))
                      (local.set $best (local.get $dk))
                      (local.set $bestj (i32.add (local.get $j) (local.get $k))))
                    (else
                      (if (i32.lt_s (local.get $dk) (local.get $second))
                        (then (local.set $second (local.get $dk))))))
                  (local.set $k (i32.add (local.get $k) (i32.const 1)))
                  (br_if $lanes (i32.lt_u (local.get $k) (i32.const 4))))))
            (local.set $j (i32.add (local.get $j) (i32.const 4)))
            (local.set $p (i32.add (local.get $p) (i32.const 128)))
            (br $train4)))
        ;; the rest, one at a time
        (block $end1
          (loop $train1
            (br_if $end1 (i32.ge_u (local.get $j) (local.get $nb)))
            (local.set $s0 (call $dist (local.get $q0) (local.get $q1) (local.get $p)))
            (local.set $dk (i32.add
              (i32.add (i32x4.extract_lane 0 (local.get $s0)) (i32x4.extract_lane 1 (local.get $s0)))
              (i32.add (i32x4.extract_lane 2 (local.get $s0)) (i32x4.extract_lane 3 (local.get $s0)))))
            (if (i32.lt_s (local.get $dk) (local.get $best))
              (then
                (local.set $second (local.get $best))
                (local.set $best (local.get $dk))
                (local.set $bestj (local.get $j)))
              (else
                (if (i32.lt_s (local.get $dk) (local.get $second))
                  (then (local.set $second (local.get $dk))))))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (local.set $p (i32.add (local.get $p) (i32.const 32)))
            (br $train1)))
        (i32.store (i32.add (local.get $out) (i32.mul (local.get $i) (i32.const 12))) (local.get $bestj))
        (i32.store offset=4 (i32.add (local.get $out) (i32.mul (local.get $i) (i32.const 12))) (local.get $best))
        (i32.store offset=8 (i32.add (local.get $out) (i32.mul (local.get $i) (i32.const 12))) (local.get $second))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $queries))))
)
