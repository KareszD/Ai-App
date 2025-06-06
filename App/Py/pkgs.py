from pathlib import Path
import os
import sysconfig

# Let cwd be .../App/Py (where you ran python)
# project_root becomes .../App
project_root = Path(os.getcwd())

# Now create a Path object for requirements.txt
reqs_path = project_root / "requirements.txt"

print("Project root:", project_root)
print("requirements.txt path:", reqs_path)


pkgs = []
if reqs_path.exists():
    for line in reqs_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name = line.split('==')[0].split('>=')[0].split('>')[0]
        pkgs.append(name)

# 3) hiddenimports = all submodules of each pkg
hidden_imports = []
stdlib_path = Path(sysconfig.get_paths()['stdlib'])  # e.g. "C:\...\Lib" on Windows, or "/usr/lib/python3.10" on Linux
encodings_dir = stdlib_path / 'encodings'
datas = []

# Copy every file under that encodings/ folder into the frozen app
for f in encodings_dir.rglob('*'):
    if f.is_file():
        # src = the real file on disk
        src = str(f.resolve())
        # dst = the path relative to stdlib_path, e.g. "encodings/utf_8.py"
        dst = str(f.relative_to(stdlib_path))
        datas.append((src, dst))

print(datas)