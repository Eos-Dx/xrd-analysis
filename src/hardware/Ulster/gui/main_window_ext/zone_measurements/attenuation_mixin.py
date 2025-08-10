from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class AttenuationMixin:
    """
    UI mixin for the Attenuation tab in Zone Measurements.
    This is a minimal implementation to satisfy imports and provide
    a placeholder tab. Extend with real logic as needed.
    """

    def create_attenuation_tab(self):
        # Ensure self.tabs exists
        if not hasattr(self, "tabs") or self.tabs is None:
            return
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Attenuation measurements UI goes here."))
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Attenuation")
