from wht_visualizer.wht_visualizer import WHTVisualizer
import sys

def test_simple_window():
    print(" -> [Test] Creating Visualizer instance...")
    viz = WHTVisualizer(title="Simple Integrity Test", show=True)
    
    print(" -> [Test] Calling show() - Expecting window to appear.")
    viz.show()
    print(" -> [Test] Window closed by user.")

if __name__ == "__main__":
    test_simple_window()
