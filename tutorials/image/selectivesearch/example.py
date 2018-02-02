#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: example.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2018-01-05 22:46:21
############################

# slove Error:  _tkinter.TclError: no display name and no $DISPLAY environment variable
# 在脚本导入任何库之前，运行：
import matplotlib
matplotlib.use('Agg')

# begin
import skimage.data
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch


def main():

    # loading astronaut image
    # img = skimage.data.astronaut()
    img = skimage.io.imread('test1.jpg')
    #transform.resize(img, (256, 256))
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()
    plt.savefig('test1.png')

if __name__ == "__main__":
    main()


