from datetime import datetime

from .Sample import Sample


class FakeLiveSample(Sample):

    def __init__(self, data_set, name, fps):
        super().__init__(data_set, name)
        self.fps = fps
        self.last_id = -1
        self.start_time = None

    def get_image(self, img_id):
        if self.last_id == -1:
            self.start_time = datetime.now()
        time_passed = (datetime.now() - self.start_time)

        self.last_id += time_passed.seconds * self.fps
        return self.images[self.last_id]
