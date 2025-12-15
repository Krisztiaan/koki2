import json
from pathlib import Path


def test_colab_notebooks_are_clean() -> None:
    notebooks = sorted(Path("colab").glob("*.ipynb"))
    assert notebooks, "expected at least one notebook under colab/"

    dirty: list[str] = []
    for path in notebooks:
        nb = json.loads(path.read_text(encoding="utf-8"))

        if "widgets" in nb.get("metadata", {}):
            dirty.append(f"{path}: metadata.widgets present")

        for cell_i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue

            if cell.get("execution_count") is not None:
                dirty.append(f"{path}: cell[{cell_i}] has execution_count={cell.get('execution_count')}")
                break

            outputs = cell.get("outputs", [])
            if outputs:
                dirty.append(f"{path}: cell[{cell_i}] has {len(outputs)} outputs")
                break

    assert not dirty, (
        "Committed notebooks should be output-free to avoid noisy diffs.\n"
        + "\n".join(f"- {item}" for item in dirty)
        + "\n\nTip: when running on Colab, use File → Save a copy in Drive."
    )
