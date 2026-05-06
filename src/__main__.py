"""Entry point so `python -m src` runs the Actor."""

import asyncio

from src.main import main

asyncio.run(main())
