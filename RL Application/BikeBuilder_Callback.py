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
            
        # Tracking per episode
        n_envs = self.training_env.num_envs
        self.ep_steps      = np.zeros(n_envs, dtype=np.int64)
        self.ep_intersects = np.zeros(n_envs, dtype=np.int64)
        self.ep_reuses     = np.zeros(n_envs, dtype=np.int64)

        # Tracking lists for per episode collectors
        self.intersect_pcts : list[float] = []
        self.reuse_pcts     : list[float] = []

    def _log_histogram(self, tag: str, counts: np.ndarray) -> None:
        if self._tb_writer is None or counts.sum() == 0:
            return
        samples   = np.repeat(np.arange(len(counts)), counts)
        bin_edges = np.arange(len(counts) + 1) - 0.5   # one bin per category, centered on each integer
        self._tb_writer.add_histogram(tag, samples, self.num_timesteps, bins=bin_edges) # type: ignore

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, info in enumerate(infos):
            self.distance_sum += info["d_reward"]
            self.progress_sum += info["p_reward"]
            self.step_count   += 1

            self.ep_steps[i]      += 1
            self.ep_intersects[i] += int(info["ccx_count"])
            self.ep_reuses[i]     += int(info["reuse_count"])

            if dones[i]:
                self.intersect_pcts.append(100.0 * self.ep_intersects[i] / self.ep_steps[i])
                self.reuse_pcts.append(100.0 * self.ep_reuses[i] / self.ep_steps[i])
                self.ep_steps[i]      = 0
                self.ep_intersects[i] = 0
                self.ep_reuses[i]     = 0

        assert self.grammar is not None
        for action in self.locals["actions"]:
            self.grammar.record(action)

        return True

    def _on_rollout_end(self) -> None:
        if self.step_count == 0:
            return # Safeguard for 0 division
        assert self.grammar is not None                             # To fix PyLance prompt

        # ---    Rewards     ---
        self.logger.record("rewards/distance_step_score_mean", self.distance_sum / self.step_count)
        self.logger.record("rewards/progress_step_score_mean", self.progress_sum / self.step_count)

        # ---   Penalties    ---
        if self.intersect_pcts:
            self.logger.record("penalties/intersection_ep_percentage", float(np.mean(self.intersect_pcts)))
        if self.reuse_pcts:
            self.logger.record("penalties/reuse_ep_percentage", float(np.mean(self.reuse_pcts)))

        # --- Shape Grammars ---
        self._log_histogram("grammar/stock_usage",     self.grammar.stock.counts)
        self._log_histogram("grammar/target_usage",    self.grammar.target.counts)
        self._log_histogram("grammar/candidate_usage", self.grammar.candidate.counts)
        self._log_histogram("grammar/mirror_usage",    self.grammar.mirror.counts)

        # ---  Reset ---
        self.distance_sum = 0.0
        self.progress_sum = 0.0
        self.step_count   = 0
        self.grammar.reset()
        self.intersect_pcts = []
        self.reuse_pcts     = []