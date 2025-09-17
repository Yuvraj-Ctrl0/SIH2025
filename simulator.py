import threading, time
class DeviceSimulator:
    def __init__(self):
        self.light=False; self.fan=False; self.mode='AUTO'
        threading.Thread(target=self._log, daemon=True).start()
    def _log(self):
        while True:
            print(f'[DeviceSim] mode={self.mode} light={self.light} fan={self.fan}')
            time.sleep(5)
    def on_command(self, payload):
        if 'mode' in payload: self.mode = payload['mode']
        if 'lights' in payload: self.light = (payload['lights']=='ON')
        if 'fans' in payload: self.fan = (payload['fans']=='ON')
    def on_occupancy(self, state):
        if self.mode=='AUTO':
            if state=='EMPTY': self.light=False; self.fan=False
            if state=='OCCUPIED': self.light=True; self.fan=True
