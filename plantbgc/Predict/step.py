class PredictStep(object):

    def run(self, record):
        raise NotImplementedError()

    def print_summary(self):
        raise NotImplementedError()
