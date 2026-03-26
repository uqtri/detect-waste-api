import argparse
import os
import shutil
from collections import Counter


CLASS_TO_SUPERCLASS = {
    # paper
    "leaflet": "paper",
    "newspaper": "paper",
    "napkin": "paper",
    # plastic
    "plasticbag": "plastic",
    "plasticbottle": "plastic",
    "plasticene": "plastic",
    # metal
    "cans": "metal",
    "battery": "metal",
    # glass
    "glassbottle": "glass",
    "bulb": "glass",
    # food
    "bread": "food",
    "leftovers": "food",
    "watermelonrind": "food",
    "nut": "food",
    # medical
    "bandaid": "medical",
    "diapers": "medical",
    "facialmask": "medical",
    "medicinebottle": "medical",
    "tabletcapsule": "medical",
    "thermometer": "medical",
    "traditionalChinesemedicine": "medical",
    # personal care
    "toothbrush": "personal_care",
    "toothpastetube": "personal_care",
    "nailpolishbottle": "personal_care",
    # other (explicit examples from request)
    "bowlsanddishes": "other",
    "chopsticks": "other",
    "penholder": "other",
    "pesticidebottle": "other",
    "toothpick": "other",
    "XLight": "other",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remap 34 waste classes into 8 super-classes."
    )
    parser.add_argument(
        "--input-root",
        default="classifier/images_square",
        help="Root containing train/ and test/ class folders.",
    )
    parser.add_argument(
        "--output-root",
        default="classifier/images_square_8cls",
        help="Destination root where remapped train/ and test/ are created.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "move"],
        default="copy",
        help="Copy files (safe) or move files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-root before processing.",
    )
    return parser.parse_args()


def list_class_dirs(split_dir):
    if not os.path.isdir(split_dir):
        return []
    return sorted(
        d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
    )


def transfer_file(src, dst, copy_mode):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def remap_split(input_split_dir, output_split_dir, copy_mode):
    class_dirs = list_class_dirs(input_split_dir)
    if not class_dirs:
        print(f"[WARN] Split not found or empty: {input_split_dir}")
        return Counter(), Counter(), 0

    src_count = Counter()
    dst_count = Counter()
    moved_files = 0

    for cls_name in class_dirs:
        src_cls_dir = os.path.join(input_split_dir, cls_name)
        # Anything not in the explicit map goes to "other".
        super_cls = CLASS_TO_SUPERCLASS.get(cls_name, "other")

        filenames = [
            f for f in os.listdir(src_cls_dir) if os.path.isfile(os.path.join(src_cls_dir, f))
        ]
        src_count[cls_name] += len(filenames)
        dst_count[super_cls] += len(filenames)

        for fname in filenames:
            src_path = os.path.join(src_cls_dir, fname)
            dst_path = os.path.join(output_split_dir, super_cls, f"{cls_name}__{fname}")
            transfer_file(src_path, dst_path, copy_mode)
            moved_files += 1

    return src_count, dst_count, moved_files


def main():
    args = parse_args()

    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    if args.overwrite and os.path.exists(output_root):
        shutil.rmtree(output_root)

    os.makedirs(output_root, exist_ok=True)

    print(f"Input root : {input_root}")
    print(f"Output root: {output_root}")
    print(f"Mode       : {args.copy_mode}")

    total_files = 0
    for split in args.splits:
        print(f"\n=== Processing split: {split} ===")
        in_split = os.path.join(input_root, split)
        out_split = os.path.join(output_root, split)
        os.makedirs(out_split, exist_ok=True)

        src_count, dst_count, processed = remap_split(
            in_split, out_split, args.copy_mode
        )
        total_files += processed

        if processed == 0:
            continue

        print("Source class distribution:")
        for k, v in sorted(src_count.items()):
            print(f"  {k}: {v}")

        print("Mapped class distribution:")
        for k, v in sorted(dst_count.items()):
            print(f"  {k}: {v}")

    print(f"\nDone. Processed files: {total_files}")


if __name__ == "__main__":
    main()
