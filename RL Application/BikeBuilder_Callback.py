# =============================================================================
# IMPORTS
# =============================================================================
from typing import Any
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger    import TensorBoardOutputFormat

from environment.envs.BikeBuilder_Classes import PointDict, ShapeGrammar, EpisodeGrammar
 
# =============================================================================
# CUSTOM CALLBACK
# =============================================================================

class BikeBuilder_Callback(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        # Reward accumulators
        self.distance_sum = 0.0
        self.progress_sum = 0.0
        self.step_count   = 0.0

        # Penalty accumulators
        self.reuse_sum    = 0.0
        self.ccx_sum      = 0.0

        # Shape Grammar
        self.grammar: EpisodeGrammar | None = None

    def _on_training_start(self) -> None:                    
        self._tb_writer = None
        for output_format in self.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                self._tb_writer = output_format.writer
                
            n_frames = len(self.training_env.get_attr("frame_stock", indices=0)[0])
            self.grammar = EpisodeGrammar(
                stock     = ShapeGrammar(size=n_frames),
                target    = ShapeGrammar(size=len(PointDict), labels=[p.name for p in PointDict]),
                candidate = ShapeGrammar(size=len(PointDict), labels=[p.name for p in PointDict]),
                mirror    = ShapeGrammar(size=2, labels=["original", "mirrored"]),
        )

    def _log_histogram(self, tag: str, counts: np.ndarray) -> None:
        if self._tb_writer is None or counts.sum() == 0:
            return
        samples   = np.repeat(np.arange(len(counts)), counts)
        bin_edges = np.arange(len(counts) + 1) - 0.5   # one bin per category, centered on each integer
        self._tb_writer.add_histogram(tag, samples, self.num_timesteps, bins=bin_edges) # type: ignore

    def _on_step(self) -> bool:

        for info in self.locals["infos"]:
            self.distance_sum += info["d_reward"]
            self.progress_sum += info["p_reward"]
            self.reuse_sum    += int(info["reuse_count"])
            self.ccx_sum      += int(info["ccx_count"])
            self.step_count   += 1

        assert self.grammar is not None   
        for action in self.locals["actions"]:
            self.grammar.record(action)

        return True

    def _on_rollout_end(self) -> None:
        if self.step_count == 0:
            return # Safeguard for 0 division
        assert self.grammar is not None                             # To fix PyLance prompt

        # ---    Rewards     ---
        self.logger.record("rewards/distance_mean", self.distance_sum / self.step_count)
        self.logger.record("rewards/progression mean", self.progress_sum / self.step_count)

        # ---   Penalties    ---
        self.logger.record("penalties/reuse_count", self.reuse_sum)
        self.logger.record("penalties/intersection_count", self.ccx_sum)

        # --- Shape Grammars ---
        self._log_histogram("grammar/stock_usage",     self.grammar.stock.counts)
        self._log_histogram("grammar/target_usage",    self.grammar.target.counts)
        self._log_histogram("grammar/candidate_usage", self.grammar.candidate.counts)
        self._log_histogram("grammar/mirror_usage",    self.grammar.mirror.counts)

        # ---  Reset ---
        self.distance_sum = 0.0
        self.progress_sum = 0.0
        self.reuse_sum    = 0
        self.ccx_sum      = 0
        self.step_count   = 0
        self.grammar.reset()