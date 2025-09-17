class MqttBridge:
    def __init__(self, on_status=None):
        self.on_status = on_status
    def publish_cmd(self, payload):
        print('MQTT publish:', payload)
