import signal

class TrainingSignal:

    def __init__(self):
        self.stop_training = False


    def handler(self, sig, frame):

        self.stop_training = True



training_signal = TrainingSignal()
    
signal.signal(
    signal.SIGTERM,
    training_signal.handler
)

signal.signal(
    signal.SIGINT,
    training_signal.handler
)

