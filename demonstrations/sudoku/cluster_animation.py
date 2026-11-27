"""Render recorded cluster-and-GAC deductions; optionally verify them without DFS."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont


SIZE, CELL, LEFT, TOP, SIDE = (1180, 870), 76, 44, 132, 762
COLORS = dict(background="#f7f9fc", paper="#ffffff", ink="#1c2940", muted="#65748a",
              grid="#cad2dd", heavy="#3a485f", given="#edf0f5", blue="#2670cd",
              blue_fill="#edf5ff", purple="#8554bf", red="#bd3747", red_fill="#fff0f2",
              green="#157a49", green_fill="#e5f6ec")


def _integer(value, low, high):
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def _domains(value, name):
    if not isinstance(value, list) or len(value) != 81:
        raise ValueError(f"{name} must contain 81 nonempty digit domains")
    for domain in value:
        if (not isinstance(domain, list) or not domain
                or any(not _integer(digit, 1, 9) for digit in domain)
                or len(domain) != len(set(domain))):
            raise ValueError(f"{name} contains an invalid digit domain")
    return [tuple(sorted(domain)) for domain in value]


def validate_trace(data):
    """Validate displayed states, their monotonicity, givens and change counts."""
    if data.get("version") != 1:
        raise ValueError("Expected a version-1 cluster animation trace")
    givens = data.get("givens")
    if (not isinstance(givens, list) or len(givens) != 81
            or any(not _integer(digit, 0, 9) for digit in givens)):
        raise ValueError("givens must contain 81 digits, with 0 for blanks")
    initial = _domains(data.get("initial"), "initial")
    baseline = _domains(data.get("baseline"), "baseline")
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Expected at least one recorded useful cluster")
    states = [initial, baseline]
    for index, step in enumerate(steps, 1):
        after = _domains(step.get("after"), f"step {index}")
        before = states[-1]
        removed = sum(len(old) - len(new) for old, new in zip(before, after))
        assigned = sum(len(old) > 1 and len(new) == 1 for old, new in zip(before, after))
        if step.get("primary_removed") != removed or step.get("new_assignments") != assigned:
            raise ValueError(f"Step {index} change counts differ from its displayed states")
        keys = step.get("core_keys")
        if (not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys)
                or len(keys) != len(set(keys))):
            raise ValueError(f"Step {index} must identify distinct original tensor keys")
        if (not isinstance(step.get("open_colors"), list)
                or any(not isinstance(color, str) for color in step["open_colors"])):
            raise ValueError(f"Step {index} must identify its open variables")
        for link in step.get("links", []):
            if (link.get("kind") not in ("cell", "row", "column", "box")
                    or not _integer(link.get("cell"), 0, 80)
                    or not _integer(link.get("digit"), 1, 9)
                    or (link["kind"] != "cell" and not _integer(link.get("unit"), 1, 9))):
                raise ValueError(f"Step {index} contains an invalid displayed tensor link")
        for output in step.get("outputs", []):
            kind = output.get("kind")
            if (kind not in ("cell", "indicator", "row", "column", "box")
                    or (kind in ("cell", "indicator") and not _integer(output.get("cell"), 0, 80))
                    or (kind != "cell" and not _integer(output.get("digit"), 1, 9))
                    or (kind in ("row", "column", "box") and not _integer(output.get("unit"), 1, 9))
                    or not isinstance(output.get("label"), str)):
                raise ValueError(f"Step {index} contains an invalid retained variable")
        states.append(after)
    for index, state in enumerate(states):
        if any(given and domain != (given,) for given, domain in zip(givens, state)):
            raise ValueError(f"State {index} does not preserve the givens")
        if index and any(not set(new) <= set(old) for old, new in zip(states[index - 1], state)):
            raise ValueError(f"State {index} adds a previously eliminated candidate")
    if any(len(domain) != 1 for domain in states[-1]):
        raise ValueError("The final recorded board must be solved without branching")
    return states


def _describe_color(color):
    prefix, *parts = color.split("_")
    values = list(map(int, parts))
    if prefix in ("pos", "a"):
        rb, ri, cb, ci = values[:4]
        cell = (3 * rb + ri) * 9 + 3 * cb + ci
        result = dict(kind="cell" if prefix == "pos" else "indicator", cell=cell,
                      label=f"r{3 * rb + ri + 1}c{3 * cb + ci + 1}")
        if prefix == "a":
            result.update(digit=values[4] + 1, label=result["label"] + f" = {values[4] + 1} (indicator)")
        return result
    digit, major, minor = values
    kind = {"row": "row", "col": "column", "square": "box"}[prefix]
    unit = 3 * major + minor + 1
    return dict(kind=kind, digit=digit + 1, unit=unit,
                label=f"{kind} {unit}: digit {digit + 1} positions")


def verify_trace(data):
    """Reconstruct original tensors and replay every recorded cluster plus GAC.

    This needs the installed project, but no archived dataset or checkpoint.
    Full-domain hashes include the 972 auxiliary variables, beyond the cells
    drawn in the GIF. Unit highlights are checked against actual core scopes.
    """
    import numpy as np
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
    from experiments.cluster_selection.data import network_data
    from experiments.cluster_selection.supervised.collect import sudoku_domains
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network

    states = validate_trace(data)
    network = ConstraintNetwork(get_sudoku_constraint_network(3), maintain_indices=True)
    digest = sha256(json.dumps(network_data(network), separators=(",", ":")).encode()).hexdigest()
    if digest != data["provenance"]["network_sha256"]:
        raise ValueError("The original tensor network differs from the recorded network")
    for core in network.coresDict.values():
        core.values = np.asarray(core.values, dtype=bool)
    originals = {key: (tuple(core.colors), core.values.copy()) for key, core in network.coresDict.items()}
    board = "".join(str(digit) if digit else "." for digit in data["givens"])
    state = network.create_mac_state(sudoku_domains(board))

    def forbid(*args, **kwargs):
        raise AssertionError("Branching is forbidden when verifying this trace")

    state.assign = network.solve_mac = forbid

    def check(expected, expected_hash):
        domains = state.snapshot_domains()
        primary = [tuple(value + 1 for value in domains[f"pos_{r // 3}_{r % 3}_{c // 3}_{c % 3}"])
                   for r in range(9) for c in range(9)]
        digest = sha256(json.dumps({color: list(domains[color]) for color in sorted(domains)},
                                   separators=(",", ":")).encode()).hexdigest()
        if primary != expected or digest != expected_hash:
            raise ValueError("Replayed domains differ from the recorded deduction")

    check(states[0], data["initial_sha256"])
    if not state.propagate():
        raise ValueError("Initial GAC found a contradiction")
    check(states[1], data["baseline_sha256"])
    for index, step in enumerate(data["steps"]):
        links = []
        for key in step["core_keys"]:
            colors = originals[key][0]
            atom = next(color for color in colors if color.startswith("a_"))
            category = next(color for color in colors if color != atom)
            indicator, unit = _describe_color(atom), _describe_color(category)
            link = dict(kind=unit["kind"], cell=indicator["cell"], digit=indicator["digit"])
            if unit["kind"] != "cell":
                link["unit"] = unit["unit"]
            links.append(link)
        if (links != step["links"]
                or [_describe_color(color) for color in step["open_colors"]] != step["outputs"]):
            raise ValueError("Displayed links or retained variables differ from the selected tensors")
        before = state.snapshot_domains()
        candidate = SummaryClusterCandidate(step["candidate_key"], tuple(step["core_keys"]), tuple(step["open_colors"]))
        prepared = runtime._prepare(state, candidate, **data["limits"])
        if prepared is None or prepared[0].work != step["work"] or prepared[0].max_size != step["max_size"]:
            raise ValueError("The cluster no longer matches its recorded contraction budget")
        runtime.apply_candidate(state, candidate, prepared)
        removed = sum(len(before[color]) - len(state.domains[color]) for color in before)
        if removed <= 0 or removed != step["primary_removed"] + step["auxiliary_removed"]:
            raise ValueError("The cluster does not reproduce its recorded useful pruning")
        check(states[index + 2], step["after_sha256"])
        if (state.contradiction or state.decision_level or state._added_core_keys
                or state._queue or state._queued or set(state._cores) != set(originals)
                or any(tuple(state._cores[key].colors) != colors or not np.array_equal(state._cores[key].values, values)
                       for key, (colors, values) in originals.items())):
            raise AssertionError("Verification changed original cores or retained a temporary summary")
    assignment = state.assignment()
    if not state.is_solved() or any(not values[tuple(assignment[color] for color in colors)]
                                    for colors, values in originals.values()):
        raise ValueError("The final assignment does not satisfy every original tensor")
    return dict(clusters=len(data["steps"]), variables=len(state.dimensions), original_cores=len(originals),
                original_tensors_satisfied=True, branching=False, retained_summaries=False)


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for candidate in (Path("/usr/share/fonts/truetype/dejavu") / name, name):
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            pass
    return ImageFont.load_default()


FONTS = {name: _font(size, bold) for name, size, bold in (
    ("title", 32, True), ("heading", 26, True), ("body", 19, False), ("small", 17, False),
    ("label", 17, True), ("candidate", 19, False), ("number", 42, True), ("metric", 33, True))}


def _text(draw, xy, text, font="body", color="ink", anchor=None):
    draw.text(xy, str(text), font=FONTS[font], fill=COLORS.get(color, color), anchor=anchor)


def _wrap(draw, text, x, y, width=350, font="body", color="muted", gap=6):
    line = ""
    line_height = getattr(FONTS[font], "size", 19) + gap
    for word in str(text).split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=FONTS[font]) > width:
            _text(draw, (x, y), line, font, color)
            y += line_height
            line = word
        else:
            line = trial
    if line:
        _text(draw, (x, y), line, font, color)
        y += line_height
    return y


def _rect(cell, inset=0):
    row, column = divmod(cell, 9)
    x, y = LEFT + column * CELL, TOP + row * CELL
    return (x + inset, y + inset, x + CELL - inset, y + CELL - inset)


def _point(cell, digit):
    x, y, _, _ = _rect(cell)
    row, column = divmod(digit - 1, 3)
    return x + (column + .5) * CELL / 3, y + (row + .5) * CELL / 3


def _unit_rect(kind, unit, inset=4):
    index = unit - 1
    if kind == "row":
        bounds = (LEFT, TOP + index * CELL, LEFT + 9 * CELL, TOP + (index + 1) * CELL)
    elif kind == "column":
        bounds = (LEFT + index * CELL, TOP, LEFT + (index + 1) * CELL, TOP + 9 * CELL)
    else:
        row, column = divmod(index, 3)
        bounds = (LEFT + column * 3 * CELL, TOP + row * 3 * CELL,
                  LEFT + (column + 1) * 3 * CELL, TOP + (row + 1) * 3 * CELL)
    x0, y0, x1, y1 = bounds
    return x0 + inset, y0 + inset, x1 - inset, y1 - inset


def _focus_digit(step, before, after):
    links = Counter(link["digit"] for link in step.get("links", []))
    removed = Counter(digit for old, new in zip(before, after) for digit in set(old) - set(new))
    return max(links, key=lambda digit: (removed[digit], links[digit], -digit)) if links else None


def _retained_labels(outputs):
    """List every output, grouping position variables for the same unit."""
    labels, seen = [], set()
    for output in outputs:
        kind = output["kind"]
        if kind in ("row", "column", "box"):
            key = kind, output["unit"]
            if key in seen:
                continue
            seen.add(key)
            digits = sorted({item["digit"] for item in outputs
                             if item["kind"] == kind and item.get("unit") == output["unit"]})
            word = "digit" if len(digits) == 1 else "digits"
            labels.append(f"{kind} {output['unit']}: {word} {', '.join(map(str, digits))} positions")
        else:
            labels.append(output["label"])
    return labels


def _frame(data, domains, *, phase, before=None, after=None, step=None, index=0, focus=None):
    image = Image.new("RGB", SIZE, COLORS["background"])
    draw = ImageDraw.Draw(image)
    _text(draw, (LEFT, 22), "Learned clusters, verified deductions", "title")
    _text(draw, (LEFT, 69), f"{data.get('method', 'Learned constructor')}  ·  contraction + GAC  ·  no branching",
          "body", "muted")
    for value in range(9):
        _text(draw, (LEFT + (value + .5) * CELL, TOP - 20), value + 1, "small", "muted", "mm")
        _text(draw, (LEFT - 21, TOP + (value + .5) * CELL), value + 1, "small", "muted", "mm")
    changed = {cell for cell in range(81) if before and after and before[cell] != after[cell]}
    newly_fixed = {cell for cell in changed if len(before[cell]) > 1 and len(after[cell]) == 1}
    selected = {(link["cell"], link["digit"]) for link in (step or {}).get("links", [])
                if link["digit"] == focus}
    for cell, domain in enumerate(domains):
        fill = "given" if data["givens"][cell] else "paper"
        if phase == "update" and cell in changed:
            fill = "green_fill" if cell in newly_fixed else "blue_fill"
        elif phase == "remove" and cell in changed:
            fill = "red_fill"
        draw.rectangle(_rect(cell), fill=COLORS[fill])
        if len(domain) == 1:
            x0, y0, x1, y1 = _rect(cell)
            color = "ink" if data["givens"][cell] else "green"
            _text(draw, ((x0 + x1) / 2, (y0 + y1) / 2 - 1), domain[0], "number", color, "mm")
        else:
            for digit in domain:
                removed = phase == "remove" and digit not in after[cell]
                color = "red" if removed else "blue" if (cell, digit) in selected else "muted"
                x, y = _point(cell, digit)
                if (cell, digit) in selected and not removed:
                    draw.ellipse((x - 11, y - 12, x + 11, y + 12), fill=COLORS["blue_fill"])
                _text(draw, (x, y), digit, "candidate", color, "mm")
                if removed:
                    draw.line((x - 8, y + 7, x + 8, y - 7), fill=COLORS["red"], width=2)
    for offset in range(10):
        width = 3 if offset % 3 == 0 else 1
        color = COLORS["heavy" if width == 3 else "grid"]
        draw.line((LEFT, TOP + offset * CELL, LEFT + 9 * CELL, TOP + offset * CELL), fill=color, width=width)
        draw.line((LEFT + offset * CELL, TOP, LEFT + offset * CELL, TOP + 9 * CELL), fill=color, width=width)
    units = {(link["kind"], link["unit"]) for link in (step or {}).get("links", [])
             if link["digit"] == focus and link["kind"] != "cell"}
    for kind, unit in sorted(units):
        draw.rounded_rectangle(_unit_rect(kind, unit), radius=5, outline=COLORS["blue"], width=2)
    outputs = (step or {}).get("outputs", [])
    retained_units = {(output["kind"], output["unit"]) for output in outputs
                      if output["kind"] in ("row", "column", "box")}
    for kind, unit in sorted(retained_units):
        draw.rounded_rectangle(_unit_rect(kind, unit, 8), radius=5,
                               outline=COLORS["purple"], width=2)
    for output in outputs:
        kind = output["kind"]
        if kind == "cell":
            draw.rounded_rectangle(_rect(output["cell"], 6), radius=5, outline=COLORS["purple"], width=3)
        elif kind == "indicator":
            x, y = _point(output["cell"], output["digit"])
            draw.ellipse((x - 12, y - 13, x + 12, y + 13), outline=COLORS["purple"], width=2)

    phase_title = dict(initial="Initial puzzle", baseline="Ordinary GAC", focus="Select a cluster",
                       remove="Contract, then propagate", update="Keep the new domains", solved="Solved at the root")
    tag = "START" if phase == "initial" else "GAC" if phase == "baseline" else "SOLVED" if phase == "solved" else f"CLUSTER {index} / {len(data['steps'])}"
    draw.rounded_rectangle((SIDE, TOP, SIDE + 350, TOP + 34), radius=8, fill=COLORS["blue_fill"])
    _text(draw, (SIDE + 12, TOP + 7), tag, "label", "blue")
    y = _wrap(draw, phase_title[phase], SIDE, TOP + 52, font="heading", color="ink") + 12
    assigned = sum(len(domain) == 1 for domain in domains)
    _text(draw, (SIDE, y), f"{assigned} / 81", "metric", "green" if phase == "solved" else "ink")
    _text(draw, (SIDE + 174, y + 13), "cells assigned", "small", "muted")
    y += 62
    if step:
        _text(draw, (SIDE, y), f"{len(step['core_keys'])} tensors  ·  {len(step['open_colors'])} outputs", "body")
        y += 34
        change = "Contract this cluster" if phase == "focus" else f"−{step['primary_removed']} candidates"
        _text(draw, (SIDE, y), change, "heading", "ink" if phase == "focus" else "red")
        y += 38
        change = "Then propagate the summary" if phase == "focus" else f"+{step['new_assignments']} new assignments"
        _text(draw, (SIDE, y), change, "body", "muted" if phase == "focus" else "green")
        y += 37
        if focus is not None:
            y = _wrap(draw, f"Blue: digit {focus} links. Purple: all retained variables.", SIDE, y, font="small")
        labels = _retained_labels(outputs)
        if labels:
            y += 10
            _text(draw, (SIDE, y), "Retained variables", "label", "purple")
            y += 27
            for label in labels:
                y = _wrap(draw, label, SIDE, y, font="small", color="ink", gap=3)
        y += 12
        _wrap(draw, "After contraction + GAC, discard the summary and retain its pruning.",
              SIDE, y, font="small")
    else:
        descriptions = dict(
            initial="Small digits list every value still possible in an unassigned cell. Dark numbers are the given clues.",
            baseline="First propagate the original tensor constraints. These deductions happen before the first learned cluster.",
            solved="Every cell is assigned, using propagation alone. Each useful cluster was contracted under the current domains.")
        _wrap(draw, descriptions[phase], SIDE, y, font="body")
    legend_y = TOP + 598
    draw.line((SIDE, legend_y - 18, SIDE + 350, legend_y - 18), fill=COLORS["grid"], width=1)
    for offset, (color, label) in enumerate((("blue", "Selected tensor links"), ("purple", "All retained variables"),
                                            ("red", "Candidates removed"), ("green", "Inferred assignments"))):
        y = legend_y + offset * 27
        draw.rounded_rectangle((SIDE, y + 3, SIDE + 13, y + 16), radius=3, fill=COLORS[color])
        _text(draw, (SIDE + 23, y), label, "small", "muted")
    return image


def render_trace(data, output, *, preview=None, force=False):
    """Write a deterministic GIF and optionally a representative removal frame."""
    output = Path(output)
    preview = Path(preview) if preview else None
    if preview is not None and preview.resolve() == output.resolve():
        raise ValueError("GIF and preview paths must differ")
    for path in (output, preview):
        if path and path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {path}; use --force explicitly")
    states = validate_trace(data)
    frames = [_frame(data, states[0], phase="initial"),
              _frame(data, states[1], phase="baseline", before=states[0], after=states[1])]
    durations = [2500, 3000]
    step_time = max(2400, min(9000, round(29000 / len(data["steps"]) / 10) * 10))
    preview_frame, most_removed = None, -1
    for index, step in enumerate(data["steps"], 1):
        before, after = states[index], states[index + 1]
        focus = _focus_digit(step, before, after)
        for phase, fraction in (("focus", .28), ("remove", .44), ("update", .28)):
            domains = after if phase == "update" else before
            frame = _frame(data, domains, phase=phase, before=before, after=after,
                           step=step, index=index, focus=focus)
            frames.append(frame)
            durations.append(round(step_time * fraction / 10) * 10)
            if phase == "remove" and step["primary_removed"] > most_removed:
                preview_frame, most_removed = frame, step["primary_removed"]
    frames.append(_frame(data, states[-1], phase="solved"))
    durations.append(3500)
    # One palette across all frames prevents color changes during a static hold.
    samples = Image.new("RGB", (160 * min(8, len(frames)), 120 * ((len(frames) + 7) // 8)))
    for index, frame in enumerate(frames):
        samples.paste(frame.resize((160, 120)), ((index % 8) * 160, (index // 8) * 120))
    adaptive = samples.quantize(colors=224).getpalette()
    swatches = list(dict.fromkeys([ImageColor.getrgb(color) for color in COLORS.values()]
                                 + [tuple(adaptive[i:i + 3]) for i in range(0, 224 * 3, 3)]))
    palette = Image.new("P", (1, 1))
    palette.putpalette([channel for color in swatches for channel in color]
                       + [0] * (768 - 3 * len(swatches)))
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(output, save_all=True, append_images=indexed[1:], duration=durations,
                    loop=0, optimize=True, disposal=2)
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview_frame.save(preview)
    return dict(frames=len(frames), seconds=sum(durations) / 1000, bytes=output.stat().st_size)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Destination GIF")
    parser.add_argument("--preview", type=Path, help="Optional representative PNG frame")
    parser.add_argument("--verify", action="store_true", help="Replay the recorded clusters against the original tensors")
    parser.add_argument("--force", action="store_true", help="Explicitly allow replacing generated images")
    args = parser.parse_args(argv)
    data = json.loads(args.trace.read_text())
    verification = verify_trace(data) if args.verify else None
    result = render_trace(data, args.output, preview=args.preview, force=args.force)
    if verification is not None:
        result["verification"] = verification
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
