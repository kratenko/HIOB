from ..HiobModule import HiobModule


class FeatureSelector(HiobModule):

    def reduce_features(self, tracking, frame):
        raise NotImplementedError()