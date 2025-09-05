#!/usr/bin/env python3
import sys, os
import cv2
import numpy as np

# ---------- Pretty terminal printer (Unicode box drawing) ----------
ANSI_RESET = "\033[0m"
ANSI_BOLD  = "\033[1m"
ANSI_DIM   = "\033[2m"
ANSI_WHITE = "\033[97m"
ANSI_GREEN = "\033[32m"

def _fmt_given(v):  # givens: bold white
    return ANSI_BOLD + ANSI_WHITE + str(v) + ANSI_RESET

def _fmt_solved(v):  # solver-filled: green
    return ANSI_GREEN + str(v) + ANSI_RESET

def _fmt_empty():
    return ANSI_DIM + "." + ANSI_RESET

def print_board(board, givens_mask=None):
    """board: 9x9 ints; givens_mask: 9x9 bools (True if given)."""
    top    = "┏━━━┯━━━┯━━━┳━━━┯━━━┯━━━┳━━━┯━━━┯━━━┓"
    mid3   = "┣━━━┿━━━┿━━━╋━━━┿━━━┿━━━╋━━━┿━━━┿━━━┫"
    light  = "┠───┼───┼───╂───┼───┼───╂───┼───┼───┨"
    bottom = "┗━━━┷━━━┷━━━┻━━━┷━━━┷━━━┻━━━┷━━━┷━━━┛"
    v_heavy, v_light = "┃", "│"

    def fmt_cell(r, c):
        v = board[r][c]
        if v == 0:
            return _fmt_empty()
        if givens_mask is not None and givens_mask[r][c]:
            return _fmt_given(v)
        return _fmt_solved(v)

    def row_line(r):
        vals = [ fmt_cell(r, c) for c in range(9) ]
        part1 = f" {vals[0]} {v_light} {vals[1]} {v_light} {vals[2]} "
        part2 = f" {vals[3]} {v_light} {vals[4]} {v_light} {vals[5]} "
        part3 = f" {vals[6]} {v_light} {vals[7]} {v_light} {vals[8]} "
        return f"{v_heavy}{part1}{v_heavy}{part2}{v_heavy}{part3}{v_heavy}"

    print(top)
    for r in range(9):
        print(row_line(r))
        if r == 8:
            print(bottom)
        elif (r + 1) % 3 == 0:
            print(mid3)
        else:
            print(light)

# ---------- OCR: warp grid, split 9×9, per-cell OCR with Tesseract ----------
try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
except ImportError:
    print("Missing dependency: pytesseract. Install with: pip install pytesseract")
    sys.exit(1)

def find_grid_and_warp(image_bgr, size=450):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thr  = cv2.adaptiveThreshold(blur, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 10)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    quad = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        ap = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            quad = ap.reshape(4,2).astype(np.float32)
            break
    if quad is None:
        # fallback: whole image
        h, w = gray.shape
        quad = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)

    # order tl, tr, br, bl
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1)
    rect = np.zeros((4,2), dtype=np.float32)
    rect[0] = quad[np.argmin(s)]   # tl
    rect[2] = quad[np.argmax(s)]   # br
    rect[1] = quad[np.argmin(d)]   # tr
    rect[3] = quad[np.argmax(d)]   # bl

    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, M, (size,size))
    return warped

def ocr_cell(cell_gray):
    """Return digit 1..9 or 0 if empty/uncertain."""
    h, w = cell_gray.shape
    m = int(0.18 * w)  # avoid borders
    roi = cell_gray[m:h-m, m:w-m]
    # normalize and binarize (good for screenshots)
    _, binv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # light dilation to thicken strokes
    binv = cv2.dilate(binv, np.ones((2,2), np.uint8), 1)

    cfg = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
    data = pytesseract.image_to_data(binv, config=cfg, output_type=TesseractOutput.DICT)
    best_txt, best_conf = "", -1.0
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        try:
            conf = float(conf)
        except Exception:
            conf = -1.0
        if txt and txt.strip().isdigit() and conf > best_conf:
            best_txt, best_conf = txt.strip(), conf
    return int(best_txt[0]) if best_txt and best_conf >= 30 else 0

def parse_image_to_board(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    warped = find_grid_and_warp(img, size=450)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # mild contrast boost helps some fonts
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    step = 450 // 9
    board = [[0]*9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            cell = gray[r*step:(r+1)*step, c*step:(c+1)*step]
            board[r][c] = ocr_cell(cell)
    return board

# ---------- Backtracking Sudoku solver ----------
def find_empty(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c
    return None

def valid(board, r, c, v):
    if any(board[r][x] == v for x in range(9)): return False
    if any(board[x][c] == v for x in range(9)): return False
    br, bc = (r//3)*3, (c//3)*3
    for i in range(3):
        for j in range(3):
            if board[br+i][bc+j] == v:
                return False
    return True

def solve(board):
    pos = find_empty(board)
    if not pos:
        return True
    r, c = pos
    for v in range(1, 10):
        if valid(board, r, c, v):
            board[r][c] = v
            if solve(board):
                return True
            board[r][c] = 0
    return False

# ---------- CLI ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_and_solve.py path/to/puzzle.png")
        sys.exit(2)
    image_path = sys.argv[1]

    # Parse board from image
    parsed = parse_image_to_board(image_path)

    # Mask of givens (non-zero after OCR)
    givens = [[parsed[r][c] != 0 for c in range(9)] for r in range(9)]

    print("\nParsed board:")
    print_board(parsed, givens_mask=givens)

    # Solve
    solved = [row[:] for row in parsed]
    if not solve(solved):
        print("\nNo solution found (check OCR or puzzle validity).")
        sys.exit(1)

    print("\nSolved board:")
    print_board(solved, givens_mask=givens)

if __name__ == "__main__":
    main()