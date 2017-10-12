from .Sample import Sample


class LiveSample(Sample):

    def __init__(self, data_set, name):
        super().__init__(data_set, name)
        self.last_img = -1

    def get_image(self, img_id):
        self.last_img += 1
        return self.images[self.last_img]