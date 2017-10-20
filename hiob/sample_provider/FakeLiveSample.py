from datetime import datetime

from .Sample import Sample


class FakeLiveSample(Sample):

    def __init__(self, data_set, name, fps):
        super().__init__(data_set, name)
        self.fps = fps
        self.start_time = None

    def get_next_frame_data(self):
        if self.current_frame_id == -1:
            self.start_time = datetime.now()
        time_passed = (datetime.now() - self.start_time)

        self.current_frame_id += time_passed.seconds * self.fps

        return [
            self.get_image(self.current_frame_id),
            self.get_ground_truth(self.current_frame_id)]

    def get_image(self, img_id):
        if len(self.images) > img_id:
            return self.images[img_id]
        else:
            return self.images[-1]

    def get_ground_truth(self, gt_id):
        if len(self.ground_truth) > 0:
            if len(self.ground_truth) > gt_id:
                return self.ground_truth[gt_id]
            else:
                return self.ground_truth[-1]
        return None
