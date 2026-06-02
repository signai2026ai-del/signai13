"""
ASL Hand Landmark Templates — Anatomically Accurate
=====================================================
21 MediaPipe landmarks (x, y, z) per letter, hand-crafted from actual
ASL hand pose measurements. Coordinates are wrist-relative, normalized
so that the wrist-to-middle-MCP distance = 1.0.

MediaPipe landmark order (0-20):
  0: WRIST
  1: THUMB_CMC  2: THUMB_MCP  3: THUMB_IP   4: THUMB_TIP
  5: INDEX_MCP  6: INDEX_PIP  7: INDEX_DIP  8: INDEX_TIP
  9: MID_MCP   10: MID_PIP  11: MID_DIP   12: MID_TIP
 13: RING_MCP  14: RING_PIP 15: RING_DIP  16: RING_TIP
 17: PINKY_MCP 18: PINKY_PIP 19: PINKY_DIP 20: PINKY_TIP
"""

import numpy as np

LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# ─── Raw templates: each is 21×3 float array (x, y, z) ──────────────────────
# Coordinate system (wrist-centered, normalized):
#   x: right = +, left = -
#   y: up = -, down = +  (image coords: up means smaller y)
#   z: toward camera = - (depth, mostly unused for 2D)
#
# Key reference distances (normalized to mid-MCP span = 1.0):
#   MCP row y ≈ -0.70 (above wrist)
#   PIP row y ≈ -1.10 (extended) or -0.90 (curled)
#   Fingertip extended y ≈ -1.60
#   Fingertip curled y ≈ -0.70 (touching palm)

_T = {

# ── A: Fist, thumb raised BESIDE index — thumb tip is clearly ABOVE knuckle row
#     Key distinction vs S: thumb is beside/above (not over) the finger line;
#     vs E: fingers are tightly fisted (tips touching palm), not just bent.
'A': np.array([
    [ 0.00,  0.00,  0.00],   # 0 WRIST
    [-0.30, -0.38,  0.00],   # 1 THUMB_CMC
    [-0.42, -0.58,  0.02],   # 2 THUMB_MCP
    [-0.36, -0.78,  0.02],   # 3 THUMB_IP
    [-0.28, -0.95,  0.02],   # 4 THUMB_TIP — raised up, beside index MCP
    [-0.15, -0.70,  0.00],   # 5 INDEX_MCP
    [-0.14, -0.84,  0.12],   # 6 INDEX_PIP  — tightly curled inward
    [-0.12, -0.75,  0.18],   # 7 INDEX_DIP
    [-0.10, -0.66,  0.20],   # 8 INDEX_TIP  — tucked into palm
    [ 0.02, -0.72,  0.00],   # 9 MID_MCP
    [ 0.03, -0.85,  0.12],   # 10 MID_PIP
    [ 0.03, -0.76,  0.18],   # 11 MID_DIP
    [ 0.03, -0.67,  0.20],   # 12 MID_TIP
    [ 0.18, -0.70,  0.00],   # 13 RING_MCP
    [ 0.18, -0.83,  0.12],   # 14 RING_PIP
    [ 0.18, -0.74,  0.18],   # 15 RING_DIP
    [ 0.18, -0.65,  0.20],   # 16 RING_TIP
    [ 0.33, -0.66,  0.00],   # 17 PINKY_MCP
    [ 0.33, -0.77,  0.10],   # 18 PINKY_PIP
    [ 0.33, -0.69,  0.14],   # 19 PINKY_DIP
    [ 0.33, -0.62,  0.16],   # 20 PINKY_TIP
], dtype=np.float32),

# ── B: All fingers straight up, thumb folded ────────────────────────────────
'B': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.35, -0.45,  0.00],
    [-0.20, -0.55,  0.00],
    [-0.10, -0.65,  0.00],   # thumb folded inward
    [-0.15, -0.70,  0.00],
    [-0.15, -1.00,  0.00],
    [-0.15, -1.25,  0.00],
    [-0.15, -1.50,  0.00],   # index tip — fully extended
    [ 0.02, -0.72,  0.00],
    [ 0.02, -1.02,  0.00],
    [ 0.02, -1.28,  0.00],
    [ 0.02, -1.55,  0.00],   # middle tip
    [ 0.18, -0.70,  0.00],
    [ 0.18, -1.00,  0.00],
    [ 0.18, -1.26,  0.00],
    [ 0.18, -1.52,  0.00],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.94,  0.00],
    [ 0.33, -1.16,  0.00],
    [ 0.33, -1.38,  0.00],
], dtype=np.float32),

# ── C: Curved C-shape — fingers curved, thumb out opposite ──────────────────
'C': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.40, -0.30,  0.10],
    [-0.50, -0.50,  0.10],
    [-0.55, -0.72,  0.05],
    [-0.55, -0.90,  0.00],   # thumb tip pointing right
    [-0.12, -0.70,  0.00],
    [-0.20, -0.95,  0.05],
    [-0.25, -1.10, -0.05],
    [-0.28, -1.22, -0.10],
    [ 0.02, -0.72,  0.00],
    [-0.02, -0.98,  0.05],
    [-0.05, -1.14, -0.05],
    [-0.07, -1.27, -0.10],
    [ 0.18, -0.70,  0.00],
    [ 0.15, -0.96,  0.05],
    [ 0.13, -1.11, -0.05],
    [ 0.11, -1.24, -0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.31, -0.90,  0.05],
    [ 0.30, -1.04, -0.05],
    [ 0.29, -1.15, -0.10],
], dtype=np.float32),

# ── D: Index up, rest curled, thumb touching middle ─────────────────────────
'D': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.35,  0.00],
    [-0.25, -0.55,  0.05],
    [-0.10, -0.65,  0.05],
    [ 0.02, -0.72,  0.00],   # thumb tip touching middle PIP
    [-0.15, -0.70,  0.00],
    [-0.15, -1.00,  0.00],
    [-0.15, -1.28,  0.00],
    [-0.15, -1.55,  0.00],   # index fully extended
    [ 0.02, -0.72,  0.00],
    [ 0.04, -0.85,  0.05],
    [ 0.05, -0.78,  0.10],
    [ 0.05, -0.70,  0.10],   # middle curled
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.82,  0.05],
    [ 0.18, -0.74,  0.10],
    [ 0.18, -0.66,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── E: All fingers bent into palm, thumb bent ───────────────────────────────
'E': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.32,  0.00],
    [-0.32, -0.50, -0.05],
    [-0.20, -0.60, -0.08],
    [-0.10, -0.68, -0.08],   # thumb tip
    [-0.15, -0.70,  0.00],
    [-0.12, -0.88, -0.08],
    [-0.10, -0.80, -0.12],
    [-0.09, -0.73, -0.12],
    [ 0.02, -0.72,  0.00],
    [ 0.04, -0.90, -0.08],
    [ 0.05, -0.82, -0.12],
    [ 0.05, -0.75, -0.12],
    [ 0.18, -0.70,  0.00],
    [ 0.19, -0.88, -0.08],
    [ 0.19, -0.80, -0.12],
    [ 0.19, -0.73, -0.12],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.80, -0.08],
    [ 0.33, -0.73, -0.10],
    [ 0.33, -0.67, -0.10],
], dtype=np.float32),

# ── F: Index and thumb form circle, others spread ───────────────────────────
'F': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.30, -0.48,  0.05],
    [-0.20, -0.60,  0.05],
    [-0.12, -0.70,  0.00],   # thumb tip touching index
    [-0.15, -0.70,  0.00],
    [-0.14, -0.80,  0.05],
    [-0.13, -0.72,  0.08],
    [-0.12, -0.65,  0.08],   # index curled to touch thumb
    [ 0.02, -0.72,  0.00],
    [ 0.02, -1.02,  0.00],
    [ 0.02, -1.28,  0.00],
    [ 0.02, -1.52,  0.00],   # middle extended
    [ 0.18, -0.70,  0.00],
    [ 0.18, -1.00,  0.00],
    [ 0.18, -1.26,  0.00],
    [ 0.18, -1.50,  0.00],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.94,  0.00],
    [ 0.33, -1.16,  0.00],
    [ 0.33, -1.38,  0.00],
], dtype=np.float32),

# ── G: Index and thumb pointing sideways (pistol shape) ─────────────────────
'G': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.50, -0.45,  0.00],
    [-0.65, -0.50,  0.00],
    [-0.80, -0.55,  0.00],   # thumb pointing left horizontally
    [-0.15, -0.70,  0.00],
    [-0.35, -0.72,  0.00],
    [-0.55, -0.74,  0.00],
    [-0.75, -0.76,  0.00],   # index pointing left
    [ 0.02, -0.72,  0.00],
    [ 0.04, -0.85,  0.05],
    [ 0.04, -0.77,  0.10],
    [ 0.04, -0.70,  0.10],
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.82,  0.05],
    [ 0.18, -0.74,  0.10],
    [ 0.18, -0.66,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── H: Index and middle pointing sideways ────────────────────────────────────
'H': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.42, -0.48,  0.00],
    [-0.28, -0.60,  0.00],
    [-0.18, -0.70,  0.00],
    [-0.15, -0.70,  0.00],
    [-0.35, -0.72,  0.00],
    [-0.55, -0.74,  0.00],
    [-0.75, -0.76,  0.00],   # index horizontal
    [ 0.02, -0.72,  0.00],
    [-0.18, -0.74,  0.00],
    [-0.38, -0.76,  0.00],
    [-0.58, -0.78,  0.00],   # middle horizontal
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.82,  0.05],
    [ 0.18, -0.74,  0.10],
    [ 0.18, -0.66,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── I: Pinky up, others fisted ───────────────────────────────────────────────
'I': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.40, -0.48,  0.05],
    [-0.28, -0.60,  0.05],
    [-0.18, -0.70,  0.05],
    [-0.15, -0.70,  0.00],
    [-0.15, -0.82,  0.05],
    [-0.15, -0.74,  0.10],
    [-0.15, -0.66,  0.10],
    [ 0.02, -0.72,  0.00],
    [ 0.02, -0.83,  0.05],
    [ 0.02, -0.75,  0.10],
    [ 0.02, -0.67,  0.10],
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.81,  0.05],
    [ 0.18, -0.73,  0.10],
    [ 0.18, -0.65,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.96,  0.00],
    [ 0.33, -1.22,  0.00],
    [ 0.33, -1.48,  0.00],   # pinky fully extended
], dtype=np.float32),

# ── J: Like I but pinky angled (motion letter; static = I-variant) ───────────
'J': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.40, -0.48,  0.05],
    [-0.28, -0.60,  0.05],
    [-0.18, -0.70,  0.05],
    [-0.15, -0.70,  0.00],
    [-0.15, -0.82,  0.05],
    [-0.15, -0.74,  0.10],
    [-0.15, -0.66,  0.10],
    [ 0.02, -0.72,  0.00],
    [ 0.02, -0.83,  0.05],
    [ 0.02, -0.75,  0.10],
    [ 0.02, -0.67,  0.10],
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.81,  0.05],
    [ 0.18, -0.73,  0.10],
    [ 0.18, -0.65,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.40, -0.90,  0.00],   # pinky PIP — angled outward (J-shape)
    [ 0.48, -1.10,  0.00],
    [ 0.56, -1.30,  0.00],   # pinky tip angled right
], dtype=np.float32),

# ── K: Index up, middle angled, thumb between ────────────────────────────────
'K': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.28, -0.35,  0.00],
    [-0.18, -0.52,  0.05],
    [-0.05, -0.62,  0.05],
    [ 0.05, -0.72,  0.05],   # thumb between index and middle
    [-0.15, -0.70,  0.00],
    [-0.15, -1.00,  0.00],
    [-0.15, -1.28,  0.00],
    [-0.15, -1.55,  0.00],   # index extended
    [ 0.02, -0.72,  0.00],
    [ 0.10, -0.98,  0.00],
    [ 0.18, -1.18,  0.00],
    [ 0.26, -1.38,  0.00],   # middle angled
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.82,  0.05],
    [ 0.18, -0.74,  0.10],
    [ 0.18, -0.66,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── L: L-shape — index up, thumb horizontal ──────────────────────────────────
'L': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.50, -0.45,  0.00],
    [-0.65, -0.50,  0.00],
    [-0.80, -0.55,  0.00],   # thumb pointing far left
    [-0.15, -0.70,  0.00],
    [-0.15, -1.00,  0.00],
    [-0.15, -1.28,  0.00],
    [-0.15, -1.55,  0.00],   # index straight up
    [ 0.02, -0.72,  0.00],
    [ 0.02, -0.83,  0.05],
    [ 0.02, -0.75,  0.10],
    [ 0.02, -0.67,  0.10],
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.81,  0.05],
    [ 0.18, -0.73,  0.10],
    [ 0.18, -0.65,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── M: Three fingers folded over thumb ───────────────────────────────────────
'M': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.32, -0.30,  0.05],
    [-0.22, -0.48,  0.08],
    [-0.10, -0.58,  0.08],
    [ 0.00, -0.65,  0.08],   # thumb under fingers
    [-0.15, -0.70,  0.00],
    [-0.14, -0.82, -0.08],
    [-0.13, -0.76, -0.12],
    [-0.12, -0.70, -0.12],
    [ 0.02, -0.72,  0.00],
    [ 0.03, -0.84, -0.08],
    [ 0.04, -0.78, -0.12],
    [ 0.04, -0.72, -0.12],
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.82, -0.08],
    [ 0.18, -0.76, -0.12],
    [ 0.18, -0.70, -0.12],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],   # pinky out
], dtype=np.float32),

# ── N: Two fingers over thumb ─────────────────────────────────────────────────
'N': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.32, -0.30,  0.05],
    [-0.22, -0.48,  0.08],
    [-0.10, -0.58,  0.08],
    [ 0.00, -0.65,  0.08],
    [-0.15, -0.70,  0.00],
    [-0.14, -0.82, -0.08],
    [-0.13, -0.76, -0.12],
    [-0.12, -0.70, -0.12],
    [ 0.02, -0.72,  0.00],
    [ 0.03, -0.84, -0.08],
    [ 0.04, -0.78, -0.12],
    [ 0.04, -0.72, -0.12],
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.81,  0.05],
    [ 0.18, -0.73,  0.10],
    [ 0.18, -0.65,  0.10],   # ring extended (unlike M)
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── O: All fingers curved to meet thumb (O-shape) ────────────────────────────
'O': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.38, -0.32,  0.00],
    [-0.38, -0.52,  0.05],
    [-0.28, -0.68,  0.05],
    [-0.16, -0.80,  0.00],   # thumb tip meeting fingertips
    [-0.15, -0.70,  0.00],
    [-0.16, -0.90,  0.05],
    [-0.15, -0.82,  0.02],
    [-0.15, -0.74,  0.00],   # index curved to O
    [ 0.02, -0.72,  0.00],
    [ 0.01, -0.93,  0.05],
    [ 0.00, -0.84,  0.02],
    [-0.01, -0.76,  0.00],
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.90,  0.05],
    [ 0.17, -0.82,  0.02],
    [ 0.16, -0.74,  0.00],
    [ 0.33, -0.66,  0.00],
    [ 0.32, -0.85,  0.05],
    [ 0.31, -0.77,  0.02],
    [ 0.30, -0.70,  0.00],
], dtype=np.float32),

# ── P: Like K but hand tilted (pointing down) ────────────────────────────────
'P': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.28, -0.30,  0.00],
    [-0.15, -0.45,  0.05],
    [-0.02, -0.55,  0.05],
    [ 0.10, -0.65,  0.05],
    [-0.10, -0.65,  0.00],
    [-0.05, -0.85,  0.00],
    [ 0.00, -1.05,  0.00],
    [ 0.05, -1.25,  0.00],   # index down-angled
    [ 0.05, -0.68,  0.00],
    [ 0.14, -0.88,  0.00],
    [ 0.23, -1.05,  0.00],
    [ 0.32, -1.22,  0.00],   # middle down-angled
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.82,  0.05],
    [ 0.18, -0.74,  0.10],
    [ 0.18, -0.66,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── Q: Index and thumb pointing down ─────────────────────────────────────────
'Q': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.30, -0.28,  0.00],
    [-0.20, -0.42,  0.00],
    [-0.08, -0.52,  0.00],
    [ 0.04, -0.62,  0.00],   # thumb tip
    [-0.10, -0.65,  0.00],
    [-0.05, -0.80,  0.00],
    [ 0.00, -0.95,  0.00],
    [ 0.05, -1.10,  0.00],   # index pointing downward
    [ 0.02, -0.72,  0.00],
    [ 0.02, -0.83,  0.05],
    [ 0.02, -0.75,  0.10],
    [ 0.02, -0.67,  0.10],
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.81,  0.05],
    [ 0.18, -0.73,  0.10],
    [ 0.18, -0.65,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── R: Index and middle crossed ───────────────────────────────────────────────
'R': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.40, -0.48,  0.05],
    [-0.28, -0.60,  0.05],
    [-0.18, -0.70,  0.05],
    [-0.15, -0.70,  0.00],
    [-0.08, -0.98,  0.05],
    [-0.02, -1.22,  0.05],
    [ 0.04, -1.46,  0.05],   # index crossed slightly right
    [ 0.02, -0.72,  0.00],
    [-0.04, -1.00,  0.05],
    [-0.10, -1.24,  0.05],
    [-0.16, -1.48,  0.05],   # middle crossed left
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.81,  0.05],
    [ 0.18, -0.73,  0.10],
    [ 0.18, -0.65,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── S: Fist, thumb crosses IN FRONT of the fingers (negative z = forward)
#     Key vs A: thumb goes across/over fingertips (z=-0.10..-0.16), not beside them.
#     Key vs E: all fingertips touch/tuck into palm, thumb wraps across them.
'S': np.array([
    [ 0.00,  0.00,  0.00],   # 0 WRIST
    [-0.35, -0.30,  0.00],   # 1 THUMB_CMC
    [-0.32, -0.50, -0.04],   # 2 THUMB_MCP  — comes forward
    [-0.16, -0.64, -0.10],   # 3 THUMB_IP   — crosses in front
    [ 0.00, -0.74, -0.16],   # 4 THUMB_TIP  — over fingernails (negative z = forward)
    [-0.15, -0.70,  0.00],   # 5 INDEX_MCP
    [-0.13, -0.82,  0.08],   # 6 INDEX_PIP  — fingers curled
    [-0.11, -0.74,  0.14],   # 7 INDEX_DIP
    [-0.09, -0.66,  0.18],   # 8 INDEX_TIP  — tucked
    [ 0.02, -0.72,  0.00],   # 9 MID_MCP
    [ 0.03, -0.82,  0.08],   # 10 MID_PIP
    [ 0.04, -0.74,  0.14],   # 11 MID_DIP
    [ 0.04, -0.67,  0.18],   # 12 MID_TIP
    [ 0.18, -0.70,  0.00],   # 13 RING_MCP
    [ 0.18, -0.81,  0.08],   # 14 RING_PIP
    [ 0.18, -0.73,  0.14],   # 15 RING_DIP
    [ 0.18, -0.65,  0.18],   # 16 RING_TIP
    [ 0.33, -0.66,  0.00],   # 17 PINKY_MCP
    [ 0.33, -0.77,  0.06],   # 18 PINKY_PIP
    [ 0.33, -0.70,  0.10],   # 19 PINKY_DIP
    [ 0.33, -0.63,  0.14],   # 20 PINKY_TIP
], dtype=np.float32),

# ── T: Thumb pushes BETWEEN index and middle MCPs, index PIP curls OVER thumb
#     Key vs A: thumb is sandwiched between index/middle (not beside them).
#     Key vs S: thumb tip is between fingers (positive z), not crossing over.
#     Key feature: thumb MCP is inside the knuckle gap.
'T': np.array([
    [ 0.00,  0.00,  0.00],   # 0 WRIST
    [-0.30, -0.32,  0.00],   # 1 THUMB_CMC
    [-0.20, -0.50,  0.06],   # 2 THUMB_MCP  — pushes up between fingers
    [-0.06, -0.60,  0.08],   # 3 THUMB_IP   — inside the gap
    [ 0.04, -0.68,  0.10],   # 4 THUMB_TIP  — between index and middle MCP (positive z)
    [-0.15, -0.70,  0.00],   # 5 INDEX_MCP
    [-0.14, -0.84,  0.06],   # 6 INDEX_PIP  — curled over the thumb
    [-0.10, -0.77,  0.10],   # 7 INDEX_DIP  — hooked inward
    [-0.06, -0.70,  0.12],   # 8 INDEX_TIP  — pressing on thumb
    [ 0.02, -0.72,  0.00],   # 9 MID_MCP
    [ 0.03, -0.84,  0.06],   # 10 MID_PIP   — curled (pressed against thumb side)
    [ 0.04, -0.76,  0.10],   # 11 MID_DIP
    [ 0.04, -0.68,  0.12],   # 12 MID_TIP
    [ 0.18, -0.70,  0.00],   # 13 RING_MCP
    [ 0.18, -0.82,  0.06],   # 14 RING_PIP
    [ 0.18, -0.74,  0.10],   # 15 RING_DIP
    [ 0.18, -0.66,  0.12],   # 16 RING_TIP
    [ 0.33, -0.66,  0.00],   # 17 PINKY_MCP
    [ 0.33, -0.76,  0.05],   # 18 PINKY_PIP
    [ 0.33, -0.69,  0.08],   # 19 PINKY_DIP
    [ 0.33, -0.63,  0.10],   # 20 PINKY_TIP
], dtype=np.float32),

# ── U: Index and middle up together ───────────────────────────────────────────
'U': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.40, -0.48,  0.05],
    [-0.28, -0.60,  0.05],
    [-0.18, -0.70,  0.05],
    [-0.15, -0.70,  0.00],
    [-0.15, -1.00,  0.00],
    [-0.15, -1.28,  0.00],
    [-0.15, -1.55,  0.00],   # index up
    [ 0.02, -0.72,  0.00],
    [ 0.02, -1.02,  0.00],
    [ 0.02, -1.28,  0.00],
    [ 0.02, -1.55,  0.00],   # middle up (adjacent to index)
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.81,  0.05],
    [ 0.18, -0.73,  0.10],
    [ 0.18, -0.65,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── V: Index and middle spread (victory/peace sign) ───────────────────────────
'V': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.40, -0.48,  0.05],
    [-0.28, -0.60,  0.05],
    [-0.18, -0.70,  0.05],
    [-0.25, -0.70,  0.00],
    [-0.30, -1.00,  0.00],
    [-0.33, -1.28,  0.00],
    [-0.36, -1.55,  0.00],   # index angled left
    [ 0.12, -0.72,  0.00],
    [ 0.18, -1.02,  0.00],
    [ 0.22, -1.28,  0.00],
    [ 0.26, -1.55,  0.00],   # middle angled right (spread)
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.81,  0.05],
    [ 0.18, -0.73,  0.10],
    [ 0.18, -0.65,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── W: Index, middle, ring spread ─────────────────────────────────────────────
'W': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.40, -0.48,  0.05],
    [-0.28, -0.60,  0.05],
    [-0.18, -0.70,  0.05],
    [-0.28, -0.70,  0.00],
    [-0.34, -1.00,  0.00],
    [-0.38, -1.28,  0.00],
    [-0.42, -1.55,  0.00],   # index left
    [ 0.02, -0.72,  0.00],
    [ 0.02, -1.02,  0.00],
    [ 0.02, -1.28,  0.00],
    [ 0.02, -1.55,  0.00],   # middle center
    [ 0.28, -0.70,  0.00],
    [ 0.32, -1.00,  0.00],
    [ 0.35, -1.26,  0.00],
    [ 0.38, -1.52,  0.00],   # ring right (spread)
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],   # pinky down
], dtype=np.float32),

# ── X: Index hooked/crooked ───────────────────────────────────────────────────
'X': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.40, -0.48,  0.05],
    [-0.28, -0.60,  0.05],
    [-0.18, -0.70,  0.05],
    [-0.15, -0.70,  0.00],
    [-0.10, -0.95,  0.00],
    [-0.05, -1.10, -0.10],
    [-0.12, -1.05, -0.15],   # index hooked (curled tip)
    [ 0.02, -0.72,  0.00],
    [ 0.02, -0.83,  0.05],
    [ 0.02, -0.75,  0.10],
    [ 0.02, -0.67,  0.10],
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.81,  0.05],
    [ 0.18, -0.73,  0.10],
    [ 0.18, -0.65,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

# ── Y: Thumb and pinky extended, others curled ───────────────────────────────
'Y': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.52, -0.45,  0.00],
    [-0.68, -0.50,  0.00],
    [-0.84, -0.55,  0.00],   # thumb extended far left
    [-0.15, -0.70,  0.00],
    [-0.15, -0.82,  0.05],
    [-0.15, -0.74,  0.10],
    [-0.15, -0.66,  0.10],
    [ 0.02, -0.72,  0.00],
    [ 0.02, -0.83,  0.05],
    [ 0.02, -0.75,  0.10],
    [ 0.02, -0.67,  0.10],
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.81,  0.05],
    [ 0.18, -0.73,  0.10],
    [ 0.18, -0.65,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.45, -0.86,  0.00],
    [ 0.58, -1.06,  0.00],
    [ 0.70, -1.26,  0.00],   # pinky extended right-downward
], dtype=np.float32),

# ── Z: Index pointing, drawing Z motion (static: index out angled) ──────────
'Z': np.array([
    [ 0.00,  0.00,  0.00],
    [-0.35, -0.30,  0.00],
    [-0.40, -0.48,  0.05],
    [-0.28, -0.60,  0.05],
    [-0.18, -0.70,  0.05],
    [-0.15, -0.70,  0.00],
    [-0.05, -0.95,  0.00],
    [ 0.08, -1.15,  0.00],
    [ 0.22, -1.35,  0.00],   # index pointing diagonally upper-right
    [ 0.02, -0.72,  0.00],
    [ 0.02, -0.83,  0.05],
    [ 0.02, -0.75,  0.10],
    [ 0.02, -0.67,  0.10],
    [ 0.18, -0.70,  0.00],
    [ 0.18, -0.81,  0.05],
    [ 0.18, -0.73,  0.10],
    [ 0.18, -0.65,  0.10],
    [ 0.33, -0.66,  0.00],
    [ 0.33, -0.74,  0.05],
    [ 0.33, -0.68,  0.08],
    [ 0.33, -0.62,  0.08],
], dtype=np.float32),

}


def get_template(letter: str) -> np.ndarray:
    """Return the 21×3 landmark template for a letter."""
    return _T[letter.upper()].copy()


def get_features(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Flatten landmarks to 63-feature vector, normalized relative to wrist.
    Returns: (63,) float32 array
    """
    pts = landmarks_21x3.copy().astype(np.float32)
    wrist = pts[0].copy()
    pts -= wrist
    span = np.linalg.norm(pts[9])  # MIDDLE_MCP
    if span > 1e-6:
        pts /= span
    return pts.flatten()


def generate_training_sample(letter: str, noise_level: float = 0.04) -> np.ndarray:
    """Generate one augmented 63-feature sample for a letter."""
    template = get_template(letter)

    # Gaussian noise on each joint
    noise = np.random.normal(0, noise_level, template.shape).astype(np.float32)

    # Random 2D rotation in XY plane
    angle = np.random.uniform(-0.25, 0.25)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    template = template @ rot.T + noise

    # Random global scale variation
    scale = np.random.uniform(0.85, 1.15)
    template = template * scale

    # Random translation
    shift = np.random.normal(0, 0.05, (1, 3)).astype(np.float32)
    template = template + shift

    return get_features(template)


def build_dataset(samples_per_class: int = 800, noise_levels=None):
    """
    Generate a synthetic ASL landmark dataset.
    Returns: (X, y) where X is (N, 63) and y is (N,) int labels.
    """
    if noise_levels is None:
        noise_levels = [0.02, 0.04, 0.06, 0.08]

    X_list, y_list = [], []
    for label_idx, letter in enumerate(LETTERS):
        per_level = samples_per_class // len(noise_levels)
        for noise in noise_levels:
            for _ in range(per_level):
                feat = generate_training_sample(letter, noise_level=noise)
                X_list.append(feat)
                y_list.append(label_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


if __name__ == '__main__':
    X, y = build_dataset(samples_per_class=10)
    print(f'Dataset shape: X={X.shape}, y={y.shape}')
    print(f'Features: min={X.min():.3f} max={X.max():.3f}')
    # Verify all 26 letters present
    unique = sorted(set(y.tolist()))
    print(f'Classes: {len(unique)} of 26')
