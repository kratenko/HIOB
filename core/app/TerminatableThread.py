import threading


class TerminatableThread(threading.Thread):
    terminating = False

    def stop(self):
        self.terminating = True
