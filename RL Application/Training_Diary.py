# =============================================================================
# IMPORTS
# =============================================================================
import dataclasses
import inspect
import json
import time
 
from datetime import datetime
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from internal_variables import IV

# =============================================================================
# INTROSPECTION HELPERS
# =============================================================================

# target  : actual class (PPO, BikeFrame)
# provided: dict that has been passed to the class (the env_kwargs / policy_kwargs - not every parameter available)
# resolved: complete dictionary containing both specified and unspecified/default kwargs
def resolve_kwargs(target, provided: dict) -> dict:
    # Reads the constructors of a dictionary
    # Ensures default values are also recorded

    sig      = inspect.signature(target)
    resolved = {}

    for name, param in sig.parameters.items():
        if name == "self": # Defensive guard so the 'self' item is not documented - though inspect.signature shouldn't in the first place
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue  # skip *args / **kwargs entries themselves

        if name in provided:
            resolved[name] = provided[name]
        elif param.default is not inspect.Parameter.empty:
            resolved[name] = param.default
        else:
            resolved[name] = "<required, not provided>" #Defensive guard - Though the training would crash if this occurs

    return resolved

def sanitize(value):
    # Converts values that can't be written to JSON as-is (numpy arrays,
    # lists of custom objects, class references, file paths) into short
    # descriptive summaries. Plain scalars pass through untouched.
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
 
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
 
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
 
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return f"{type(value).__name__} len=0"
        if isinstance(value[0], (str, int, float, bool)):
            return list(value)  # plain list of numbers/strings - keep as-is
        return f"{type(value).__name__}[{type(value[0]).__name__}] len={len(value)}"
 
    if isinstance(value, Path):
        return str(value)
 
    if isinstance(value, type):
        return value.__name__  # a class reference, e.g. features_extractor_class
 
    return f"<{type(value).__name__}>"  # fallback for anything else

def capture_iv() -> dict:
    # Pulls every field currently on IV, automatically - a new field added to
    # InternalVariables shows up here with no change needed in this file.
    raw = dataclasses.asdict(IV)
    return {k: sanitize(v) for k, v in raw.items()}

# =============================================================================
# TRAINING DIARY
# =============================================================================
 
class TrainingDiary:
    '''
    Records one JSON line per training run to a central log file.
    Usage:
        diary = TrainingDiary()
        diary.start(run_id=timestamp, env_class=BikeBuilder_Env, env_kwargs=env_kwargs,
                     model_class=PPO, model_kwargs=model_kwargs)
        model.learn(...)
        diary.finish()
    '''
 
    def __init__(self, diary_path: str = "training_diary.jsonl"):
        # Defaults to sitting next to this file, regardless of which script
        # imports it or what folder it's run from.
        self.diary_path = Path(__file__).resolve().parent / diary_path
        self._record: dict = {}
        self._start_time: float = 0.0
 
    def start(
            self,
            run_id      : str,
            env_class,
            env_kwargs  : dict,
            model_class,
            model_kwargs: dict,
            callback_class  = None,
            callback_kwargs : dict | None = None,
            note            : str | None = None,
        ) -> None:
 
        note = input("Describe the goal of this run: ")
        if note is None:
            note = input("Describe the goal of this run: ")
 
        self._record = {
            "run_id"     : run_id,
            "note"       : note,
            "started_at" : datetime.now().isoformat(timespec="seconds"),
 
            "internal_variables": capture_iv(),
 
            "env_class"  : env_class.__name__,
            "env_kwargs" : {k: sanitize(v) for k, v in resolve_kwargs(env_class, env_kwargs).items()},
 
            "model_class" : model_class.__name__,
            "model_kwargs": {k: sanitize(v) for k, v in resolve_kwargs(model_class, model_kwargs).items()},
        }

        if callback_class is not None:
            self._record["callback_class"]  = callback_class.__name__
            self._record["callback_kwargs"] = {
                k: sanitize(v) for k, v in resolve_kwargs(callback_class, callback_kwargs or {}).items()
        }

        self._start_time = time.perf_counter()
 
    def finish(self) -> None:
        duration = time.perf_counter() - self._start_time
        self._record["duration_seconds"] = round(duration, 2)
        self._record["finished_at"]      = datetime.now().isoformat(timespec="seconds")
 
        with open(self.diary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._record, default=str) + "\n")
 
        print(f"Training diary entry written to {self.diary_path}")

# =============================================================================
# LOADING FOR COMPARISON
# =============================================================================
 
def load_diary(diary_path: str = "training_diary.jsonl") -> pd.DataFrame:
    # Reads the whole diary file into one DataFrame - one row per run.
    # pandas fills in blanks automatically for runs that predate a given field,
    # so nothing here needs to change as the schema grows over time.
    path = Path(__file__).resolve().parent / diary_path
    return pd.read_json(path, lines=True, dtype={"run_id": str})

# =============================================================================
# RECORD LOOKUP
# =============================================================================

def get_run(run_id: str, diary_path: str = "training_diary.jsonl") -> dict:
    # Looks up one run's full record by its run_id (the timestamp printed when
    # the run started) - for pulling settings back out to repeat or tweak a run.
    df    = load_diary(diary_path)
    match = df[df["run_id"] == run_id]

    if match.empty:
        raise ValueError(f"No run found with run_id={run_id!r}")

    return match.iloc[0].to_dict()