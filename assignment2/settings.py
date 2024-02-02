import platform
from os import path
from typing import Dict

ROOT = path.abspath(path.join(__file__, ".."))

URLS: Dict[str, str] = {
    "Darwin":  "https://maxcoltheart.files.wordpress.com/2019/05/drc-mac.zip",
    'Windows': "https://maxcoltheart.files.wordpress.com/2019/05/drc-win.zip",
    'Linux':   "https://maxcoltheart.files.wordpress.com/2019/05/drc-linux-x86_64.zip"
}

URL = URLS[platform.system()]

DRC_DIR_NAME = "drc"
DRC_DIR = path.join(ROOT, DRC_DIR_NAME)

BINARY = "drc"
BINARY_PATH = path.join(DRC_DIR, BINARY)

DEFAULT_PARAMETER_FILE = "Languages/english-1.1.8.1/default.parameters"
DEFAULT_PARAMETER_PATH = path.join(DRC_DIR, DEFAULT_PARAMETER_FILE)
