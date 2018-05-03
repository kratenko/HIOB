from datetime import datetime

from .Sample import Sample


class FakeLiveSample(Sample):

    def __init__(self, data_set, name, fps):
        super().__init__(data_set, name)
        self.fps = float(fps)
        self.start_time = None
        self.current_frame_id = -1
        self.frames_processed = 0
        self.prestream_count = 100

    async def get_next_frame_data(self):
        if self.start_time is None:
            self.start_time = datetime.now()
            print("FakeLiveSample::init")
        self.frames_processed += 1
        time_passed = datetime.now() - self.start_time

        curr_frame = time_passed.total_seconds() * self.fps
        if self.prestream_count > curr_frame:
            self.current_frame_id = 0
        else:
            self.current_frame_id = int(curr_frame) - self.prestream_count
        print("{} seconds passed - current frame: {}".format(time_passed.total_seconds(), self.current_frame_id))
        return [
            self.get_image(self.current_frame_id),
            self.get_ground_truth(self.current_frame_id)]

    def get_image(self, img_id):
        if len(self.img_paths) > img_id:
            return super(FakeLiveSample, self).get_image(img_id)
            #return self.image_cache[img_id]
        else:
            return super(FakeLiveSample, self).get_image(len(self.img_paths - 1))

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
        return len(self.img_paths) - self.frames_processed
