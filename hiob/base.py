class HiobModule(object):

    def configure(self, configuration):
        raise NotImplementedError()

    def setup(self, session):
        raise NotImplementedError()
