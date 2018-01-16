from .Sample import Sample


class LiveSample(Sample):

    def __init__(self, data_set, name):
        super().__init__(data_set, name)
