from tkinter import *
import time

class Sketchpad(Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Button-1>", self.save_posn)
        self.bind("<B1-Motion>", self.add_line)

        # Stores the strokes from the user
        self.strokes = []
        
    def save_posn(self, event):
        self.lastx, self.lasty = event.x, event.y

        # Initialize start time when user starts drawing
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()
        
        # Gets the time since the last stroke in milisecond
        self.last_time = (time.time() - self.start_time) * 1000
        self.strokes.append((event.x, event.y, int(self.last_time)))
        print(self.strokes)

    def add_line(self, event):
        self.create_line((self.lastx, self.lasty, event.x, event.y))
        self.save_posn(event)

root = Tk()
root.columnconfigure(0, weight=2)
root.rowconfigure(0, weight=2)

sketch = Sketchpad(root, width = 256, height = 256)
sketch.grid(column=0, row=0, sticky=(N, W, E, S))

root.mainloop()