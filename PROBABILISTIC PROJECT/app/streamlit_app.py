"""
Root entrypoint for Streamlit application.
Runs canteen_queue/app/streamlit_app.py.
"""
import sys
from pathlib import Path

# Add current workspace root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from canteen_queue.app.streamlit_app import main

if __name__ == "__main__":
    main()
