#!/usr/bin/env python3
import cv2, numpy as np, os, sys

try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
except ImportError:
    print("Please install pytesseract: pip install pytesseract")
    sys.exit(1)

def find_grid_and_warp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    approx = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        ap = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            approx = ap.reshape(4,2).astype(np.float32)
            break
    if approx is None:
        h, w = gray.shape
        approx = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    s = approx.sum(axis=1)
    diff = np.diff(approx, axis=1)
    rect = np.zeros((4,2), dtype=np.float32)
    rect[0] = approx[np.argmin(s)] # tl
    rect[2] = approx[np.argmax(s)] # br
    rect[1] = approx[np.argmin(diff)] # tr
    rect[3] = approx[np.argmax(diff)] # bl
    size = 450
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (size,size))
    return warped

def ocr_cell(cell):
    h, w = cell.shape
    m = int(0.18*w)  # margin to avoid borders
    roi = cell[m:h-m, m:w-m]
    _, binv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    binv = cv2.dilate(binv, kernel, 1)
    cfg = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
    data = pytesseract.image_to_data(binv, config=cfg, output_type=TesseractOutput.DICT)
    best_txt, best_conf = "", -1
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        try: conf = float(conf)
        except: conf = -1
        if txt and txt.strip().isdigit() and conf > best_conf:
            best_txt, best_conf = txt.strip(), conf
    return int(best_txt[0]) if best_txt and best_conf >= 30 else 0

# Pretty terminal printer (Unicode box drawing)
ANSI_RESET = "\033[0m"
ANSI_BOLD  = "\033[1m"
ANSI_DIM   = "\033[2m"
ANSI_WHITE = "\033[97m"

def _fmt_cell(v):
    if v == 0:
        return ANSI_DIM + "." + ANSI_RESET
    return ANSI_BOLD + ANSI_WHITE + str(v) + ANSI_RESET  # highlight givens

def print_board(b):
    # Heavy outer border, heavy every 3, light inside
    top    = "┏━━━┯━━━┯━━━┳━━━┯━━━┯━━━┳━━━┯━━━┯━━━┓"
    mid3   = "┣━━━┿━━━┿━━━╋━━━┿━━━┿━━━╋━━━┿━━━┿━━━┫"
    light  = "┠───┼───┼───╂───┼───┼───╂───┼───┼───┨"
    bottom = "┗━━━┷━━━┷━━━┻━━━┷━━━┷━━━┻━━━┷━━━┷━━━┛"
    v_heavy, v_light = "┃", "│"

    def row_line(r):
        vals = [ _fmt_cell(b[r][c]) for c in range(9) ]
        # group 3 cells with light dividers, heavy between 3×3 blocks
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

def parse_image(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    warped = find_grid_and_warp(img)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g2 = clahe.apply(gray)
    cell_h = cell_w = 450//9
    board = [[0]*9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            cell = g2[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            board[r][c] = ocr_cell(cell)
    return board

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_only.py puzzle.png")
        sys.exit(1)
    board = parse_image(sys.argv[1])
    print("Parsed board:")
    print_board(board)