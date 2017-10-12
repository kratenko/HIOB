import tkinter as tk

from PIL import ImageTk


class ImageLabel(tk.Label):

    def __init__(self, *args, **kwargs):
        tk.Label.__init__(self, *args, **kwargs)
        self._image = None

    def set_image(self, image):
        if image is None:
            self._image = None
            self['image'] = None
        else:
            self._image = ImageTk.PhotoImage(image)
            self['image'] = self._image