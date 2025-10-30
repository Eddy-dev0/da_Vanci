import serial  # kommt von pyserial, später in echt benutzen

class MachineDriver:
    """
    Verantwortlich für Kommunikation mit der echten Mal-Maschine.
    -> Verbindet sich über USB/Serial
    -> Schickt Fahrbefehle, z.B. 'geh zu X,Y,Z', 'nimm Werkzeug', 'tauche in Farbe'
    """

    def __init__(self, port: str = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None  # serielle Verbindung

    def connect(self):
        """
        Baut Verbindung auf. (Noch Dummy!)
        """
        # TODO: später aktivieren
        # self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        # self.ser.flush()
        return True

    def send_command(self, cmd: str):
        """
        Schickt einen einzelnen Maschinenbefehl.
        Beispiel später: 'G1 X120 Y50 Z2 F600'
        """
        print(f"[DRIVER] -> {cmd}")
        # TODO: später über self.ser.write(...) schicken
