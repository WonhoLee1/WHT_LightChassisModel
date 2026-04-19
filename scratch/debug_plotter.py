import pyvista as pv
from pyvistaqt import BackgroundPlotter
import time
import sys

print("Step 1: Initializing BackgroundPlotter...")
try:
    # Use show=True to force immediate window creation
    plotter = BackgroundPlotter(show=True)
    print(f" -> Success! Plotter: {plotter}")
except Exception as e:
    print(f" -> Failed Plotter Init: {e}")
    sys.exit(1)

print("Step 2: Adding a sample mesh (Sphere)...")
try:
    plotter.add_mesh(pv.Sphere(), color='red')
    print(" -> Mesh added.")
except Exception as e:
    print(f" -> Failed add_mesh: {e}")

print("Step 3: Waiting 5 seconds to see if window pops up...")
for i in range(5):
    print(f" -> Waiting... {5-i}")
    time.sleep(1)

print("Step 4: Closing plotter...")
try:
    plotter.close()
    print(" -> Closed.")
except Exception as e:
    print(f" -> Failed close: {e}")

print("Final Step: Done.")
