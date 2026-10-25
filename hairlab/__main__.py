"""Command line entry point: ``python -m hairlab label``."""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

from hairlab.clipstore import ClipStore, LabelStore

# Below this, the crop holds too few real pixels of the hand for a twisted
# strand to be visible — to a model or to the person labelling it.
_FAINT_CROP_PX = 200


def cmd_label(args: argparse.Namespace) -> None:
    """Serve the labelling UI."""
    import uvicorn

    from hairlab.server.app import create_app

    store = ClipStore()
    labels = LabelStore()
    pending = len(store.clip_ids()) - len(labels.labelled_ids())

    url = f"http://{args.host}:{args.port}/"
    print(f"[hairlab] Клипы: {store.root}")
    print(f"[hairlab] Ждут разметки: {max(0, pending)}")
    print(f"[hairlab] Интерфейс: {url}")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(store, labels), host=args.host, port=args.port, log_level="warning")


def cmd_stats(_args: argparse.Namespace) -> None:
    """Print how much of the dataset is labelled, and whether it is usable."""
    store = ClipStore()
    labels = LabelStore()
    clip_ids = store.clip_ids()
    counts = labels.counts()
    done = sum(counts.values())

    print(f"Клипов всего:  {len(clip_ids)}")
    print(f"Размечено:     {done}")
    print(f"Осталось:      {max(0, len(clip_ids) - done)}")

    if counts:
        print("\nПо меткам:")
        for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {label:<12} {count}")

    # Health. Two things decide whether this dataset can train anything, and
    # both are cheap to check now and expensive to discover after two weeks:
    # a clip without landmarks has no training input at all, and the crop's
    # size in the original frame is the real detail on the hand, whatever
    # resolution it was later stored at.
    no_landmarks = faint = 0
    source_px: list[int] = []
    for clip_id in clip_ids:
        try:
            meta = store.read_meta(clip_id)
        except Exception:
            no_landmarks += 1
            continue
        frames = meta.get("frames") or []
        if not any(f.get("hand") for f in frames):
            no_landmarks += 1
        sizes = [f["source_px"] for f in frames if f.get("source_px")]
        if sizes:
            source_px.append(sorted(sizes)[len(sizes) // 2])
            if sorted(sizes)[len(sizes) // 2] < _FAINT_CROP_PX:
                faint += 1

    if clip_ids:
        print("\nКачество данных:")
        print(f"  без координат кисти (не годятся для обучения): {no_landmarks}")
        if source_px:
            source_px.sort()
            median = source_px[len(source_px) // 2]
            print(f"  детализация кисти: медиана вырезки {median} px в исходном кадре")
            print(f"  клипов с мелкой вырезкой (<{_FAINT_CROP_PX} px): {faint}")
            if median < _FAINT_CROP_PX:
                print("\n  Вырезка мелкая. Направьте камеру на голову или сядьте ближе —")
                print("  это даёт больше деталей, чем любая настройка в коде.")


# Subcommands that own their own argument parser. Delegated to before this
# module's parser runs, so each keeps its own flags and help text instead of
# having them restated — and drift between the two restatements is exactly
# why `hairlab train`, documented in hairlab/train.py since it was written,
# did not actually work.
_DELEGATED = {
    "train": "hairlab.train",
    "truncation": "hairlab.truncation",
}


def main(argv: list[str] | None = None) -> None:
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0] in _DELEGATED:
        import importlib

        module = importlib.import_module(_DELEGATED[args_in[0]])
        return module.main(args_in[1:])

    parser = argparse.ArgumentParser(prog="hairlab", description="Сбор и разметка данных для HairGuard")
    sub = parser.add_subparsers(dest="command", required=True)

    p_label = sub.add_parser("label", help="Открыть интерфейс разметки")
    p_label.add_argument("--host", default="127.0.0.1")
    p_label.add_argument("--port", type=int, default=8765)
    p_label.add_argument("--no-browser", action="store_true", help="Не открывать браузер")
    p_label.set_defaults(func=cmd_label)

    p_stats = sub.add_parser("stats", help="Показать прогресс разметки")
    p_stats.set_defaults(func=cmd_stats)

    # Listed so `hairlab --help` mentions them; the real parsing happens in
    # the delegation above.
    sub.add_parser("train", help="Обучить детектор накручивания (своя справка)",
                   add_help=False)
    sub.add_parser("truncation", help="Насколько рано можно спрашивать модель",
                   add_help=False)

    args = parser.parse_args(args_in)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
