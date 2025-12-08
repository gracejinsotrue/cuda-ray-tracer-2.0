#!/usr/bin/env python3
"""
convert ppm image format to png
"""

import sys
from PIL import Image

def convert_ppm_to_png(ppm_path, png_path=None):

    try:
        # 1) open file
        img = Image.open(ppm_path)
        
        # generate the output path
        if png_path is None:
            png_path = ppm_path.rsplit('.', 1)[0] + '.png'
        
        # saves to png
        img.save(png_path, 'PNG')
        
        print(f"successfully converted {ppm_path} to {png_path}")
        return True
        
    except FileNotFoundError:
        print(f"error: file {ppm_path}' not found")
        return False
    except Exception as e:
        print(f"error converting file: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    
    ppm_path = sys.argv[1]
    png_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_ppm_to_png(ppm_path, png_path)

if __name__ == "__main__":
    main()