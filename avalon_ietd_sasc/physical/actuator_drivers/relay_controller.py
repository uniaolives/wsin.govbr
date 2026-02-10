class RelayController:
    """
    [REVISOR VÊ]: Controlador de relé para bombas e aquecedores.
    [NÓS VEMOS]: Atuador de colapso de realidade local.
    """
    def __init__(self, id, pin):
        self.id = id
        self.pin = pin
        self.state = False

    def set_state(self, state):
        self.state = state
        print(f"   [Relay] {self.id} -> {'ON' if state else 'OFF'}")
