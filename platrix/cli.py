"""Command-line interface.

Examples
--------
    platrix serve                      # launch the web dashboard/API
    platrix run 0                      # process the default webcam
    platrix run rtsp://cam/stream      # process a network camera
    platrix run car.jpg --show         # recognize a single image
    platrix events --limit 20          # print the latest detections
"""

from __future__ import annotations

import argparse
import sys

from platrix import __version__
from platrix.config import get_settings
from platrix.core.pipeline import RecognitionPipeline, annotate
from platrix.logging_conf import configure_logging, get_logger
from platrix.sources import open_source
from platrix.storage import EventStore

logger = get_logger(__name__)


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    if args.source:
        settings.default_source = args.source
    host = args.host or settings.host
    port = args.port or settings.port
    logger.info("Starting Platrix server on http://%s:%d", host, port)
    uvicorn.run(
        "platrix.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=settings.log_level.lower(),
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = EventStore(settings)
    pipeline = RecognitionPipeline(settings)
    pipeline.warmup()

    show = args.show
    count = 0
    with open_source(args.source, loop=args.loop) as source:
        for frame in source.frames():
            readings = pipeline.process(frame)
            for reading in readings:
                if pipeline.is_duplicate(reading):
                    continue
                store.record(reading, direction=args.direction)
                count += 1
                print(
                    f"[{reading.timestamp.isoformat()}] "
                    f"plate={reading.text or '-'} score={reading.score:.2f} "
                    f"source={reading.source}"
                )
            if show:
                import cv2

                cv2.imshow("Platrix", annotate(frame.image, readings))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    if show:
        import cv2

        cv2.destroyAllWindows()
    logger.info("Done — %d plate(s) logged", count)
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    store = EventStore(get_settings())
    for ev in store.recent(limit=args.limit, plate=args.plate):
        print(
            f"{ev['created_at']}  {ev['plate_text'] or '-':<10}  "
            f"score={ev['score']:.2f}  {ev['source']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="platrix", description="Platrix ALPR engine")
    parser.add_argument("--version", action="version", version=f"Platrix {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run the web dashboard / REST API")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--source", default=None, help="Auto-start this source")
    p_serve.set_defaults(func=_cmd_serve)

    p_run = sub.add_parser("run", help="Process a source from the command line")
    p_run.add_argument("source", help="Webcam index, stream URL, video or image path")
    p_run.add_argument("--show", action="store_true", help="Show an annotated window")
    p_run.add_argument("--loop", action="store_true", help="Loop video files")
    p_run.add_argument(
        "--direction",
        choices=["entry", "exit", "unknown"],
        default="unknown",
        help="Tag detections as entry/exit (for gate/lane setups)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_events = sub.add_parser("events", help="Print recent detections")
    p_events.add_argument("--limit", type=int, default=50)
    p_events.add_argument("--plate", default=None, help="Filter by plate substring")
    p_events.set_defaults(func=_cmd_events)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging(get_settings().log_level)
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
