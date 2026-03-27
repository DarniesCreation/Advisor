#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from advisor.api.main import app
import uvicorn
if __name__ == "__main__":
	port = int(os.environ.get("PORT", 7860))
	uvicorn.run(app, host="0.0.0.0", port=port)