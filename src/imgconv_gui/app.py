"""Точка входа приложения."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    window.resize(820, 520)
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
