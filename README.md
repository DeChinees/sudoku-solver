# Sudoku Image Solver

A command-line tool that detects a Sudoku grid from an image, uses OCR to recognize the numbers, and solves the puzzle.

## Prerequisites

- Python 3.x
- Tesseract OCR
- OpenCV (cv2)
- pytesseract

## Installation

### On macOS:
```bash
# Install Tesseract using Homebrew
brew install tesseract

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install opencv-python
pip install pytesseract
```

### On Debian/Ubuntu:
```bash
# Install Tesseract
sudo apt update
sudo apt install -y tesseract-ocr

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install opencv-python
pip install pytesseract
```

## Usage

You can run the solver in two ways:

1. Using Python directly:
```bash
python sudoku_from_image.py path/to/puzzle.png
```

2. Using the shell script:
```bash
./run_sudoku.sh path/to/puzzle.png
```

## Features

- Image grid detection with perspective correction
- OCR digit recognition using Tesseract
- Backtracking Sudoku solver
- Pretty-printed output with Unicode box drawing
- Color-coded display (given numbers in white, solved numbers in green)

## Tips for Best Results

- Use clear, well-lit images of Sudoku puzzles
- Ensure the entire grid is visible in the image
- High contrast between numbers and background works best
- The puzzle should be photographed from a relatively straight-on angle

## Error Handling

The program will notify you if:
- The image file cannot be read
- Required dependencies are missing
- No valid solution is found for the puzzle
- OCR fails to recognize numbers correctly

## Output Example

The program will show:
1. The parsed board (as recognized by OCR)
2. The solved puzzle (if a solution exists)

Numbers are color-coded:
- White (bold): Original numbers from the puzzle
- Green: Solved numbers
- Dots: Empty cells in the parsed board
