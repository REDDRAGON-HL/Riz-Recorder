import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rizrec.app.main import main  # noqa: E402

if __name__ == "__main__":
    main()
