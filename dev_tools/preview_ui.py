"""Isolated browser acceptance server. Uses a saved frame and never calls ADB."""
import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Clicker, make_handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18766)
    args = parser.parse_args()
    source = np.asarray(Image.open(args.image).convert("RGB"))

    class PreviewDevice:
        def __init__(self, config, stop):
            self.config, self.stop, self.image = config, stop, None

        def screenshot(self):
            self.image = source.copy()
            return self.image

        def click(self, button):
            raise RuntimeError("UI 验收环境不发送真实点击")

        def stuck_record_add(self, button):
            pass

    with tempfile.TemporaryDirectory(prefix="mumu-ui-") as directory:
        root = Path(directory)
        shutil.copy2(ROOT / "config.json", root / "config.json")
        shutil.copytree(ROOT / "assets", root / "assets")
        shutil.copytree(ROOT / "models", root / "models")
        app = Clicker(root, PreviewDevice)
        app.capture()
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
        print(json.dumps({"url": f"http://127.0.0.1:{args.port}", "config": str(root / "config.json"),
                          "real_clicks": False}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            app.stop()
            if app.thread:
                app.thread.join(22)
            server.server_close()
            for handler in list(app.log.handlers):
                handler.close()
                app.log.removeHandler(handler)


if __name__ == "__main__":
    main()
