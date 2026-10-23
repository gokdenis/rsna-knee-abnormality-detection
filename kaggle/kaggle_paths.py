# Shared path helpers, inlined into every Kaggle kernel by scripts/build_kernels.py.
import os
import shutil
import zipfile

INPUT = "/kaggle/input"


def _input_roots():
    """Attached dataset/kernel folders, skipping the (huge) competition mounts."""
    roots = []
    for top in sorted(os.listdir(INPUT)):
        path = os.path.join(INPUT, top)
        if "rsna-knee-abnormality-detection" in top or top == "competitions":
            continue
        roots.append(path)
    return roots


def find_file(name, max_depth=6, prefer=None):
    """A file called `name` under an attached non-competition input (bounded walk).

    If several match, the first whose path contains `prefer` wins.
    """
    hits = []
    for root in _input_roots():
        base_depth = root.count(os.sep)
        for d, dirs, files in os.walk(root):
            if d.count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
            if name in files:
                hits.append(os.path.join(d, name))
                if prefer is None or prefer in d:
                    return hits[-1]
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{name} not found under {INPUT}: {os.listdir(INPUT)}")


def code_dir():
    """Directory from which `import knee` works, whatever layout Kaggle gave the code dataset."""
    try:
        src = os.path.dirname(find_file("preprocess.py", prefer="knee-code"))
    except FileNotFoundError:
        archive = find_file("knee.zip", prefer="knee-code")
        src = "/kaggle/tmp/code_zip"
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(src)
        src = os.path.dirname(find_file_under(src, "preprocess.py"))
    if os.path.basename(src) == "knee":
        return os.path.dirname(src)
    target = "/kaggle/tmp/code/knee"  # files arrived flat: rebuild the package folder
    os.makedirs(target, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith(".py"):
            shutil.copy(os.path.join(src, name), target)
    return os.path.dirname(target)


def find_file_under(root, name):
    for d, _, files in os.walk(root):
        if name in files:
            return os.path.join(d, name)
    raise FileNotFoundError(f"{name} not in {root}")


def comp_root():
    for base in [f"{INPUT}/competitions/rsna-knee-abnormality-detection",
                 f"{INPUT}/rsna-knee-abnormality-detection"]:
        if os.path.exists(os.path.join(base, "train.csv")) or os.path.exists(os.path.join(base, "test.csv")):
            return base
    raise FileNotFoundError("competition data not attached")
