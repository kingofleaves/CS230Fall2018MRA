import os
import sys
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import psutil
import tkinter

import curses

filepathExample = "/data2/yeom/ky_mra/Normal_MRA/mri_anon/N001/StanfordNormal/MRA_3DTOF_MT_-_4/IM-0001-0001-0001.dcm"

read_data = []
if len(sys.argv) == 1:
    print("argument requires filepath(s) to be loaded")
    sys.exit()

filepaths = sys.argv[1:]
for input in filepaths:
    print("Processing %s" % input)
    data = pydicom.dcmread(input)
    data_np = data.pixel_array
    read_data.append(data_np)
    plt.figure()
    plt.imshow(data_np)
# test code - read and display one dicom image (requires X-forwarding
# root = tkinter.Tk()
# frame = tkinter.Frame(root, height = 512, width = 512)
# frame.pack()
# canvas = tkinter.Canvas(frame, height = 512, width = 512)
# canvas.pack()

currIndex = 0
maxIndex = len(read_data)
print(maxIndex)
# img = plt.imshow(read_data[currIndex])
# plt.show(block = False)
plt.show()

# img = Image.fromarray(read_data[currIndex])

# photo = ImageTk.PhotoImage(image=img)
# canvas.create_image(0,0,image = photo, anchor = tkinter.NW)
# root.update()
# root.mainloop()

# img.show()


# get the curses screen window
screen = curses.initscr()
 
# turn off input echoing
curses.noecho()
 
# respond to keys immediately (don't wait for enter)
curses.cbreak()
 
# map arrow keys to special values
screen.keypad(True)
 
try:
    while True:
        char = screen.getch()
        if char == ord('q'):
            break       
        elif char == curses.KEY_UP:
            for proc in psutil.process_iter():
                if proc.name() == "display":
                    proc.kill()
            currIndex -= 1
            if currIndex < 0:
                currIndex += maxIndex
            screen.addstr(1, 0, str(currIndex)) 
            screen.addstr(2, 0, sys.argv[currIndex+1])       
            img.set_data(read_data[currIndex])
            # plt.show()

        elif char == curses.KEY_DOWN:
            for proc in psutil.process_iter():
                if proc.name() == "display":
                    proc.kill()
            currIndex += 1
            if currIndex >= maxIndex:
                currIndex -= maxIndex
            screen.addstr(1, 0, str(currIndex)) 
            screen.addstr(2, 0, sys.argv[currIndex+1]) 
            img.set_data(read_data[currIndex])
            # plt.show()

finally:
    # shut down cleanly
    curses.nocbreak(); screen.keypad(0); curses.echo()
    curses.endwin()
    sys.exit()


