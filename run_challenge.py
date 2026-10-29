"""
Run every challenge case in one go: the task-set notebooks (pz_challenge_taskset{1..4}_e2e.ipynb), which train
and save the models, then pz_challenge_results.ipynb, which scores them.

Each notebook is executed headless, from the repo root, and saved in place with its outputs, as if run by hand in
Jupyter. Cell outputs are echoed here as they come, so the progress can be followed in the terminal (or a log
file). A task set that fails is saved up to the failing cell and the others still run; the results notebook only
runs if every task set succeeded.

Settings in the notebooks' configuration cells can be overridden with --set NAME=VALUE (VALUE is Python source,
so strings need quotes). The override replaces the line that assigns NAME, so values derived from it later in the
cell (e.g. RUN from FROM_BASE) follow. The RUN of the task-set notebooks is passed on to the results notebook.

Examples (from the repo root):
    python run_challenge.py                              # task sets 1-4, then the results
    python run_challenge.py --tasksets 3 4               # task sets 3 and 4, then the results
    python run_challenge.py --no-train                   # only the results notebook
    python run_challenge.py --set ERRORS=None --set 'RUN="clean_kfold_frombase"'

On NERSC, so it survives logging out:
    nohup python run_challenge.py > run_challenge.log 2>&1 &
or inside a batch job (e.g. salloc/sbatch on a GPU node) with the pzdc env activated.
"""

import argparse
import re
import sys
import time

try:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
except ImportError:  # pragma: no cover
    sys.exit("run_challenge.py needs nbclient and nbformat (pip install nbclient nbformat)")

TASK_NOTEBOOK = "pz_challenge_taskset{}_e2e.ipynb"
RESULTS_NOTEBOOK = "pz_challenge_results.ipynb"
RUN_MARKER = "__RUN__="


class EchoClient(NotebookClient):
    """NotebookClient that also prints each cell's text output to stdout as it arrives."""

    def output(self, outs, msg, display_id, cell_index):
        content = msg["content"]
        if msg["msg_type"] == "stream":
            print(content["text"], end="", flush=True)
        elif msg["msg_type"] == "error":
            print("\n".join(content["traceback"]), flush=True)
        return super().output(outs, msg, display_id, cell_index)


def apply_overrides(nb, overrides):
    """Replace the line assigning each NAME in overrides (first code cell that has one). Returns the names found."""
    found = set()
    for name, value in overrides.items():
        pattern = re.compile(rf"^{re.escape(name)}\s*=.*$", re.MULTILINE)
        for cell in nb.cells:
            if cell.cell_type == "code" and pattern.search(cell.source):
                cell.source = pattern.sub(lambda _: f"{name} = {value}  # set by run_challenge.py", cell.source, count=1)
                found.add(name)
                break
    return found


def run_notebook(path, overrides, kernel, report_run=False):
    """Execute the notebook at path in place. Returns (ok, RUN or None)."""
    nb = nbformat.read(path, as_version=4)
    found = apply_overrides(nb, overrides)
    for name in sorted(set(overrides) - found):
        print(f"  ({path} does not set {name}; left as is)")
    if report_run:
        nb.cells.append(nbformat.v4.new_code_cell(f"print({RUN_MARKER!r} + repr(RUN))"))

    client = EchoClient(
        nb, timeout=None, kernel_name=kernel or nb.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": "."}},
    )
    ok = True
    try:
        client.execute()
    except CellExecutionError:
        ok = False

    run = None
    if report_run:
        marker_cell = nb.cells.pop()
        for out in marker_cell.get("outputs", []):
            text = out.get("text", "")
            if text.startswith(RUN_MARKER):
                run = text[len(RUN_MARKER):].strip()
    nbformat.write(nb, path)
    return ok, run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasksets", type=int, nargs="+", default=[1, 2, 3, 4], help="task sets to train")
    parser.add_argument("--no-train", action="store_true", help="skip the task-set notebooks")
    parser.add_argument("--no-results", action="store_true", help="skip the results notebook")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override a setting")
    parser.add_argument("--kernel", help="Jupyter kernel (default: the one saved in each notebook)")
    args = parser.parse_args()

    overrides = dict(item.split("=", 1) for item in args.set)
    status = {}
    run = None
    start_all = time.time()

    if not args.no_train:
        for taskset in args.tasksets:
            path = TASK_NOTEBOOK.format(taskset)
            print(f"\n#################### {path} ####################", flush=True)
            start = time.time()
            ok, run_set = run_notebook(path, overrides, args.kernel, report_run=True)
            run = run or run_set
            status[path] = f"{'ok' if ok else 'FAILED'} ({(time.time() - start) / 60:.1f} min)"

    if not args.no_results:
        if any("FAILED" in s for s in status.values()):
            status[RESULTS_NOTEBOOK] = "skipped (a task set failed)"
        else:
            results_overrides = dict(overrides)
            if run is not None and "RUN" not in overrides:
                results_overrides["RUN"] = run
            print(f"\n#################### {RESULTS_NOTEBOOK} (RUN = {results_overrides.get('RUN', 'as saved')}) "
                  "####################", flush=True)
            start = time.time()
            ok, _ = run_notebook(RESULTS_NOTEBOOK, results_overrides, args.kernel)
            status[RESULTS_NOTEBOOK] = f"{'ok' if ok else 'FAILED'} ({(time.time() - start) / 60:.1f} min)"

    print(f"\n==================== summary ({(time.time() - start_all) / 60:.1f} min) ====================")
    for path, state in status.items():
        print(f"{path}: {state}")
    sys.exit(0 if all(s.startswith("ok") for s in status.values()) else 1)


if __name__ == "__main__":
    main()
