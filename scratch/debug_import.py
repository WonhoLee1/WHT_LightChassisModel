import sys
import os

print("Step 1: Importing qtpy...")
try:
    from qtpy import QtWidgets, QtCore, QtGui
    print(f" -> qtpy imported. Backend: {getattr(QtWidgets, '__name__', 'Unknown')}")
except Exception as e:
    print(f" -> Failed qtpy: {e}")

print("Step 2: Importing qt_material...")
try:
    from qt_material import apply_stylesheet, list_themes
    print(" -> qt_material imported.")
except Exception as e:
    print(f" -> Failed qt_material: {e}")

print("Step 3: Creating QApplication instance...")
try:
    app = QtWidgets.QApplication.instance()
    if not app:
        print(" -> Creating NEW QApplication...")
        app = QtWidgets.QApplication(sys.argv)
    print(f" -> QApplication ready: {app}")
except Exception as e:
    print(f" -> Failed QApplication: {e}")

print("Final Step: Done.")
