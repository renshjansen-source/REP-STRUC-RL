''' This file generates a browsable HTML viewer for training_diary.jsonl -
    a sidebar listing every run chronologically, and a main panel showing
    the selected run's full recorded settings, colour-coded against the run
    immediately preceding it:
        green  = parameter added
        red    = parameter removed
        blue   = parameter changed
    Metadata (note, timestamps, duration) is shown separately, uncoloured,
    since it differs on every run by nature and isn't meaningful to flag.

    Usage: run this file directly - it writes training_diary_viewer.html,
    open that in a browser.
'''

# =============================================================================
# IMPORTS
# =============================================================================
import json
from pathlib import Path

from Training_Diary import load_diary

# =============================================================================
# CONSTANTS
# =============================================================================
META_KEYS = [
    "run_id", "note", "started_at", "finished_at", "duration_seconds",
    "env_class", "model_class", "callback_class",
]

# =============================================================================
# DIFF HELPERS
# =============================================================================

def _flatten(d: dict, prefix: str = "") -> dict:
    # Turns nested dicts into dot-separated leaf paths, e.g.
    # {"internal_variables": {"connection_offset": 80.0}}
    # -> {"internal_variables.connection_offset": 80.0}
    flat = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _diff(current: dict, previous: dict | None) -> dict:
    # Returns {path: "added"|"removed"|"changed"|"same"}, comparing current
    # against the immediately preceding run only - not full history.
    if previous is None:
        return {path: "same" for path in current}

    status = {}
    for path, value in current.items():
        if path not in previous:
            status[path] = "added"
        elif previous[path] != value:
            status[path] = "changed"
        else:
            status[path] = "same"
    for path in previous:
        if path not in current:
            status[path] = "removed"
    return status

# =============================================================================
# BUILD
# =============================================================================

def build_viewer(diary_path: str = "training_diary.jsonl", output_path: str = "training_diary_viewer.html") -> None:
    df = load_diary(diary_path).sort_values("started_at").reset_index(drop=True)

    runs = []
    previous_flat: dict | None = None

    for _, row in df.iterrows():
        record   = row.dropna().to_dict()
        settings = {k: v for k, v in record.items() if k not in META_KEYS}

        flat   = _flatten(settings)
        status = _diff(flat, previous_flat)

        runs.append({
            "run_id"  : record.get("run_id", "?"),
            "note"    : record.get("note", ""),
            "duration": record.get("duration_seconds", None),
            "fields"  : [
                {"path": path, "value": str(value), "status": status.get(path, "same")}
                for path, value in sorted(flat.items())
            ],
        })
        previous_flat = flat

    Path(output_path).write_text(_render_html(runs), encoding="utf-8")
    print(f"Diary viewer written to {output_path} - open it in a browser.")


def _render_html(runs: list) -> str:
    data_json = json.dumps(runs)

    return f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Training Diary Viewer</title>
<style>
  body {{ margin: 0; font-family: monospace; display: flex; height: 100vh; }}
  #sidebar {{ width: 260px; overflow-y: auto; border-right: 1px solid #ccc; background: #f7f7f7; }}
  #sidebar div {{ padding: 8px 10px; cursor: pointer; border-bottom: 1px solid #e0e0e0; font-size: 12px; }}
  #sidebar div:hover {{ background: #e8e8e8; }}
  #sidebar div.selected {{ background: #d0e4ff; }}
  #main {{ flex: 1; overflow-y: auto; padding: 16px; font-size: 13px; }}
  .added   {{ color: #1a7f37; }}
  .removed {{ color: #cf222e; text-decoration: line-through; }}
  .changed {{ color: #0969da; }}
  .same    {{ color: #24292f; }}
  .field   {{ white-space: pre; }}
  h2 {{ font-size: 15px; margin-bottom: 4px; }}
  .meta {{ color: #57606a; margin-bottom: 12px; font-size: 12px; }}
</style>
</head>
<body>
<div id="sidebar"></div>
<div id="main"></div>
<script>
const runs = {data_json};
let selectedIndex = runs.length - 1;

function renderSidebar() {{
  const sidebar = document.getElementById("sidebar");
  sidebar.innerHTML = "";
  runs.forEach((run, i) => {{
    const div = document.createElement("div");
    div.textContent = run.run_id;
    if (i === selectedIndex) div.className = "selected";
    div.onclick = () => {{ selectedIndex = i; renderSidebar(); renderMain(); }};
    sidebar.appendChild(div);
  }});
}}

function renderMain() {{
  const run  = runs[selectedIndex];
  const main = document.getElementById("main");
  main.innerHTML = "";

  const title = document.createElement("h2");
  title.textContent = run.run_id;
  main.appendChild(title);

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = (run.note ? run.note + " — " : "") +
                      (run.duration !== null ? run.duration + "s" : "");
  main.appendChild(meta);

  run.fields.forEach(f => {{
    const div = document.createElement("div");
    div.className = "field " + f.status;
    div.textContent = f.path + ": " + f.value;
    main.appendChild(div);
  }});
}}

renderSidebar();
renderMain();
</script>
</body>
</html>'''


if __name__ == "__main__":
    build_viewer()