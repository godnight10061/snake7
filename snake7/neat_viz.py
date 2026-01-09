from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Allow running this file directly (e.g. `python snake7/neat_viz.py`) by ensuring the
# repository root is on sys.path. Recommended usage is still `python -m snake7.neat_viz`.
if __package__ in (None, "") and __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    __package__ = "snake7"


def _dot_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _find_config_path(genome_path: Path) -> Path:
    config_path = genome_path.parent / "neat_config.txt"
    if config_path.exists():
        return config_path
    config_path = genome_path.parent / "snake_neat_config.txt"
    if config_path.exists():
        return config_path
    raise FileNotFoundError(f"NEAT config not found next to genome: {genome_path} (expected neat_config.txt)")


def _import_neat():
    try:
        import neat  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "neat-python is required: python -m pip install neat-python\n"
            "Or install all dev dependencies: pip install -e .[dev]"
        ) from e
    return neat


def generate_dot(genome_path: Path, config_path: Path | None = None, *, include_disabled: bool = False) -> str:
    neat = _import_neat()

    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    if config_path is None:
        config_path = _find_config_path(genome_path)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    lines = [
        "digraph neat {",
        "    rankdir=LR;",
        '    node [fontsize=9, fontname="Courier New"];',
        '    edge [fontsize=9, fontname="Courier New"];',
    ]

    input_keys = list(config.genome_config.input_keys)
    output_keys = list(config.genome_config.output_keys)
    input_index = {k: i for i, k in enumerate(input_keys)}
    output_index = {k: i for i, k in enumerate(output_keys)}

    all_node_keys = set(input_keys) | set(output_keys) | set(getattr(genome, "nodes", {}).keys())
    for u, v in getattr(genome, "connections", {}).keys():
        all_node_keys.add(int(u))
        all_node_keys.add(int(v))

    def node_sort_key(k: int) -> tuple[int, int]:
        if k in input_index:
            return (0, input_index[k])
        if k in output_index:
            return (2, output_index[k])
        return (1, int(k))

    for node_key in sorted(all_node_keys, key=node_sort_key):
        node_id = _dot_quote(str(int(node_key)))
        if node_key in input_index:
            label = f"in{input_index[node_key]}\\n{int(node_key)}"
            attrs = 'shape=box, style=filled, fillcolor=lightblue, color=blue'
        elif node_key in output_index:
            label = f"out{output_index[node_key]}\\n{int(node_key)}"
            attrs = 'shape=box, style=filled, fillcolor=mistyrose, color=red'
        else:
            label = f"h\\n{int(node_key)}"
            attrs = "shape=circle"
        lines.append(f"    {node_id} [label={_dot_quote(label)}, {attrs}];")

    for (u, v), conn in sorted(getattr(genome, "connections", {}).items(), key=lambda x: x[0]):
        enabled = bool(getattr(conn, "enabled", True))
        if not enabled and not include_disabled:
            continue

        weight = float(getattr(conn, "weight", 0.0))
        src = _dot_quote(str(int(u)))
        dst = _dot_quote(str(int(v)))

        edge_attrs: list[str] = [f"label={_dot_quote(f'{weight:+.3f}')}" ]
        if enabled:
            edge_attrs.append(f"color={'red' if weight < 0 else 'blue'}")
        else:
            edge_attrs.append("color=gray")
            edge_attrs.append("style=dotted")

        lines.append(f"    {src} -> {dst} [{', '.join(edge_attrs)}];")

    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Visualize a NEAT genome in Graphviz DOT format.")
    parser.add_argument("genome", type=Path, help="Path to the genome .pkl file.")
    parser.add_argument("--config", type=Path, default=None, help="Path to neat_config.txt (optional).")
    parser.add_argument("--out", type=Path, default=None, help="Output DOT file (default: stdout).")
    parser.add_argument("--include-disabled", action="store_true", help="Include disabled connections (dotted).")

    args = parser.parse_args()

    try:
        dot_str = generate_dot(args.genome, args.config, include_disabled=bool(args.include_disabled))
        if args.out:
            args.out.write_text(dot_str)
        else:
            print(dot_str)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
