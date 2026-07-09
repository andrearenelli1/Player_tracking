from pathlib import Path

ROOT    = Path(__file__).parent.parent
DATASET = ROOT / "dataset"

SPLITS  = ["train", "valid", "test"]


def remap_labels() -> None:
    """Remap class IDs: Ball (0) → 1, all players/referees (1-14) → 0."""
    for split in SPLITS:
        labels_dir = DATASET / split / "labels"
        for label_file in labels_dir.glob("*.txt"):
            lines = label_file.read_text().splitlines()
            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                cls = int(parts[0])
                new_cls = 1 if cls == 0 else 0   # Ball→1, person→0
                new_lines.append(" ".join([str(new_cls)] + parts[1:]))
            label_file.write_text("\n".join(new_lines) + "\n")
    print("Labels remapped.")


def fix_yaml() -> None:
    """Rewrite data.yaml with correct absolute paths and 2 classes."""
    yaml_content = f"""train: {DATASET / "train" / "images"}
val: {DATASET / "valid" / "images"}
test: {DATASET / "test" / "images"}

nc: 2
names: ['person', 'ball']
"""
    (DATASET / "data.yaml").write_text(yaml_content)
    print("data.yaml updated.")


def already_prepared() -> bool:
    yaml_path = DATASET / "data.yaml"
    if not yaml_path.exists():
        return False
    content = yaml_path.read_text()
    return "names: ['person', 'ball']" in content


def main() -> None:
    if already_prepared():
        print("Dataset already prepared — skipping to avoid double-remap.")
        return
    remap_labels()
    fix_yaml()


if __name__ == "__main__":
    main()
