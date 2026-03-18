from __future__ import annotations

import json
import os

from src.types import OrchestratorResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class JSONReporter:
    """Saves OrchestratorResult as pretty-printed JSON."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = output_dir

    def save(self, result: OrchestratorResult) -> str:
        """Write JSON report to outputs/{run_id}.json. Returns the file path."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{result.run_id}.json")

        data = result.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"JSON report saved to {path}")
        return path

    @staticmethod
    def to_json_string(result: OrchestratorResult) -> str:
        """Return the JSON string without writing to disk."""
        return json.dumps(result.to_dict(), indent=2, default=str)
