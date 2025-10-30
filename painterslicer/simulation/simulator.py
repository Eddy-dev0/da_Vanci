class Simulator:
    """
    Baut eine Vorschau (virtuelle Leinwand),
    Schritt für Schritt wie Cura's Layer View.
    """

    def __init__(self):
        pass

    def render_step(self, paint_step: dict):
        """
        Nimmt einen einzelnen Schritt aus dem Plan
        und 'malt' ihn virtuell.
        Später liefern wir hier Pixel-/Buffer-Output an die UI.
        """
        # TODO: später echtes Canvas-Rendering
        print(f"Simuliere Schritt: {paint_step}")
