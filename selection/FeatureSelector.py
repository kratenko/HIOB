from hiob import HiobModule


class FeatureSelector(HiobModule.HiobModule):

    def reduce_features(self, tracking, frame):
        raise NotImplementedError()