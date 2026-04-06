import cv2
import os
import shutil
import argparse

"""
This code is modified from a useful file provided by Zhuo Chen
"""

def figures_to_video(path_in, file_prefix, fps, delete_in=False):
    path_out = path_in + file_prefix + ".mp4"

    frame_array = []
    files = [f for f in os.listdir(path_in) 
                        if os.path.isfile(os.path.join(path_in, f)) and f.startswith(file_prefix) and f.endswith(".png")]
    files.sort(key=lambda x: int(x[x.find(" "):-4] if x.find(" ") != -1 else 0)) # sort according to the numeral before the .png
    for file in files:
        filename = path_in + file
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for frame in frame_array:
        out.write(frame)
    out.release()
    if delete_in:
        #delete all files in files
        for file in files:
            os.remove(path_in + file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert figures to video')
    parser.add_argument('path_in', type=str, help='Input path')
    parser.add_argument('file_prefix', type=str, help='File prefix')
    parser.add_argument('--fps', type=int, required = False, default = 10, help='Frames per second')
    parser.add_argument('--delete_in', dest='delete_in', action='store_true',
                        help='Whether to delete the input files after processing')
    args = parser.parse_args()

    figures_to_video(args.path_in, args.file_prefix, args.fps, args.delete_in)