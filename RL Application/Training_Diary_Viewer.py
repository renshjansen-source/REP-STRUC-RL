''' This file generates a browsable HTML viewer for training_diary.jsonl -
    a sidebar listing every run chronologically, and a main panel showing
    the selected run's full recorded settings, grouped into sections and
    colour-coded against the run immediately preceding it:
        green  = parameter added
        red    = parameter removed
        blue   = parameter changed
    Metadata (note, timestamps, duration) is shown separately, uncoloured,
    since it differs on every run by nature and isn't meaningful to flag.

    internal_variables fields are further grouped by their declared
    metadata={"section": "..."} tag (see internal_variables.py) - fields
    with no section tag fall back to "Other", so this degrades gracefully
    if that metadata hasn't been added yet.

    Usage: run this file directly - it writes training_diary_viewer.html,
    open that in a browser.
'''

# =============================================================================
# IMPORTS
# =============================================================================
import dataclasses
import json
from pathlib import Path

from internal_variables import InternalVariables
from Training_Diary import load_diary

# =============================================================================
# CONSTANTS
# =============================================================================
META_KEYS = [
    "run_id", "note", "started_at", "finished_at", "duration_seconds",
    "env_class", "model_class", "callback_class",
]

CATEGORY_ORDER = ["env_kwargs", "internal_variables", "model_kwargs", "callback_kwargs"]

# (field_name, section_name) pairs, in the same order fields are declared in
# internal_variables.py - so the viewer's subsection order matches the file's
# own grouping, rather than sorting alphabetically.
IV_FIELD_ORDER = [
    (f.name, f.metadata.get("section", "Other"))
    for f in dataclasses.fields(InternalVariables)
]

# =============================================================================
# DIFF HELPERS
# =============================================================================

def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _diff(current: dict, previous: dict | None) -> dict:
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
# SECTION BUILDING
# =============================================================================

def _make_entry(path: str, value, status: dict) -> dict:
    return {"path": path, "value": str(value), "status": status.get(path, "same")}


def _build_sections(flat: dict, status: dict) -> list[dict]:
    sections = []

    # --- env_kwargs / model_kwargs / callback_kwargs: alphabetical, no subsections ---
    for category in ["env_kwargs", "model_kwargs", "callback_kwargs"]:
        prefix  = category + "."
        entries = [
            _make_entry(path, flat[path], status)
            for path in sorted(flat.keys())
            if path.startswith(prefix)
        ]
        if entries:
            sections.append({"name": category, "subsections": [{"name": None, "fields": entries}]})

    # --- internal_variables: grouped + ordered per the dataclass definition ---
    iv_prefix = "internal_variables."
    iv_paths  = {p[len(iv_prefix):]: p for p in flat if p.startswith(iv_prefix)}

    grouped: dict[str, list] = {}
    seen = set()
    for field_name, section_name in IV_FIELD_ORDER:
        if field_name in iv_paths:
            full_path = iv_paths[field_name]
            grouped.setdefault(section_name, []).append(_make_entry(full_path, flat[full_path], status))
            seen.add(field_name)

    # IV fields present in this diary entry but no longer in the current
    # class definition (e.g. renamed or removed since that run) - keep them
    # visible rather than silently dropping them.
    for field_name, full_path in iv_paths.items():
        if field_name not in seen:
            grouped.setdefault("Other (removed)", []).append(_make_entry(full_path, flat[full_path], status))

    if grouped:
        subsections = [{"name": name, "fields": fields} for name, fields in grouped.items()]
        sections.append({"name": "internal_variables", "subsections": subsections})

    return sections

# =============================================================================
# BUILD
# =============================================================================

def build_viewer(diary_path: str = "training_diary.jsonl", output_path: str = "training_diary_viewer.html") -> None:
    df = load_diary(diary_path).sort_values("started_at", ascending=True).reset_index(drop=True)

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
            "sections": _build_sections(flat, status),
        })
        previous_flat = flat

    # Reverse for display only - diff above was already computed chronologically.
    runs.reverse()

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
  #sidebar div {{ padding: 8px 10px; cursor: pointer; border-bottom: 1px solid #e0e0e0; font-size: 15px; }}
  #sidebar div:hover {{ background: #e8e8e8; }}
  #sidebar div.selected {{ background: #d0e4ff; }}
  #main {{ flex: 1; overflow-y: auto; padding: 16px; font-size: 16px; }}
  .added   {{ color: #1a7f37; }}
  .removed {{ color: #cf222e; text-decoration: line-through; }}
  .changed {{ color: #0969da; }}
  .same    {{ color: #24292f; }}
  h2 {{ font-size: 20px; margin-bottom: 4px; }}
  .meta {{ color: #57606a; margin-bottom: 12px; font-size: 15px; }}
  .section-header {{ font-size: 18px; margin-top: 20px; margin-bottom: 6px; border-bottom: 1px solid #d0d7de; padding-bottom: 3px; }}
  .subsection-header {{ font-size: 14px; font-weight: bold; color: #57606a; margin-top: 10px; margin-bottom: 3px; }}
  .field-list {{ display: grid; grid-template-columns: max-content 1fr; column-gap: 12px; row-gap: 2px; margin-bottom: 4px; }}
  .field-row {{ display: contents; }}
  .field-label {{ white-space: pre; }}
  .field-value {{ white-space: pre-wrap; word-break: break-word; }}
</style>
</head>
<body>
<div id="sidebar"></div>
<div id="main"></div>
<script>
const runs = {data_json};
let selectedIndex = 0;

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

  run.sections.forEach(section => {{
    const sectionHeader = document.createElement("div");
    sectionHeader.className = "section-header";
    sectionHeader.textContent = section.name;
    main.appendChild(sectionHeader);

    section.subsections.forEach(sub => {{
      if (sub.name) {{
        const subHeader = document.createElement("div");
        subHeader.className = "subsection-header";
        subHeader.textContent = sub.name;
        main.appendChild(subHeader);
      }}

      const list = document.createElement("div");
      list.className = "field-list";
      sub.fields.forEach(f => {{
        const row = document.createElement("div");
        row.className = "field-row";

        const label = document.createElement("div");
        label.className = "field-label " + f.status;
        label.textContent = f.path + ":";

        const value = document.createElement("div");
        value.className = "field-value " + f.status;
        value.textContent = f.value;

        row.appendChild(label);
        row.appendChild(value);
        list.appendChild(row);
      }});
      main.appendChild(list);
    }});
  }});
}}

renderSidebar();
renderMain();
</script>
</body>
</html>'''


if __name__ == "__main__":
    build_viewer()