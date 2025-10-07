from typing import List, Dict, Any
import tinker
from tinker_atropos.interfaces import TrainingDataProcessor


class AtroposDataProcessor(TrainingDataProcessor):
    # Convert Atropos trajectories to Tinker Datum objects.
    def trajectories_to_data(
        self, trajectories: List[Any]
    ) -> tuple[List[tinker.types.Datum], Dict[str, Any]]:
        # TODO: implement
        raise NotImplementedError()
