from PySide6.QtWidgets import QApplication
import ui.main_window
import sys

def main():
    app = QApplication(sys.argv)

    window = ui.main_window.MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
