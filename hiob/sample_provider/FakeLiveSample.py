from datetime import datetime

from .Sample import Sample


class FakeLiveSample(Sample):

    def __init__(self, data_set, name, fps):
        super().__init__(data_set, name)
        self.fps = fps
        self.start_time = None
        self.current_frame_id = -1
        self.frames_processed = 0

    def get_next_frame_data(self):
        self.frames_processed += 1
        if self.current_frame_id == -1:
            self.start_time = datetime.now()
        time_passed = datetime.now() - self.start_time

        self.current_frame_id = int(time_passed.total_seconds() * self.fps)
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

    def count_frames_processed(self):
        return self.frames_processed

    def count_frames_skipped(self):
        return len(self.images) - self.frames_processed
