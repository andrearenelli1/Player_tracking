from pathlib import Path
import shutil
from PIL import Image

ROOT        = Path(__file__).parent.parent
SRC_DATASET = ROOT / "dataset"
DST_DATASET = ROOT / "dataset_ball"

SPLITS = ["train", "valid", "test"]
NATIVE_SIZE = (3840, 2160)  # keep only real match-footage frames


def already_prepared() -> bool:
    yaml_path = DST_DATASET / "data.yaml"
    if not yaml_path.exists():
        return False
    return "names: ['ball']" in yaml_path.read_text()


def build_ball_dataset() -> None:
    for split in SPLITS:
        src_imgs    = SRC_DATASET / split / "images"
        src_labels  = SRC_DATASET / split / "labels"
        dst_imgs    = DST_DATASET / split / "images"
        dst_labels  = DST_DATASET / split / "labels"

        if not src_labels.exists():
            continue

        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for label_file in src_labels.glob("*.txt"):
            # find corresponding image first — only keep native 4K frames
            img_stem = label_file.stem
            src_img = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = src_imgs / (img_stem + ext)
                if candidate.exists():
                    src_img = candidate
                    break
            if src_img is None:
                continue
            if Image.open(src_img).size != NATIVE_SIZE:
                continue

            lines = label_file.read_text().splitlines()
            ball_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                if int(parts[0]) == 1:
                    # remap class 1 (ball) → class 0 (only class in ball model)
                    ball_lines.append("0 " + " ".join(parts[1:]))

            if not ball_lines:
                continue

            # copy label
            (dst_labels / label_file.name).write_text("\n".join(ball_lines) + "\n")
            # copy image
            shutil.copy2(src_img, dst_imgs / src_img.name)

    print("Ball dataset built.")


def fix_yaml() -> None:
    yaml_content = f"""train: {DST_DATASET / "train" / "images"}
val: {DST_DATASET / "valid" / "images"}
test: {DST_DATASET / "test" / "images"}

nc: 1
names: ['ball']
"""
    (DST_DATASET / "data.yaml").write_text(yaml_content)
    print("data.yaml written.")


def main() -> None:
    if already_prepared():
        print("Ball dataset already prepared — skipping.")
        return
    build_ball_dataset()
    fix_yaml()


if __name__ == "__main__":
    main()
