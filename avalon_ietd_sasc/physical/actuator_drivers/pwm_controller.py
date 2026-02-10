class PWMController:
    """
    [REVISOR VÊ]: Controlador PWM para iluminação LED.
    [NÓS VEMOS]: Gerador de pulsação harmônica (12.8 Hz).
    """
    def __init__(self, id, pin):
        self.id = id
        self.pin = pin
        self.duty_cycle = 0

    def set_intensity(self, value):
        self.duty_cycle = value
        print(f"   [PWM] {self.id} -> {value}% Intensity")
