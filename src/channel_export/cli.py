"""Command-line interface: ``python -m src.channel_export [flags]``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from .auth import DEFAULT_TOKEN, AuthSetupError, build_services, get_credentials
from .export import Exporter, ExportOptions

log = logging.getLogger("src.channel_export")


def _iso_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from err
    if parsed > date.today():
        raise argparse.ArgumentTypeError(f"{value} is in the future")
    return parsed


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the exporter."""
    parser = argparse.ArgumentParser(
        prog="python -m src.channel_export",
        description=(
            "Export everything the YouTube Data, Analytics and Reporting APIs expose for your own "
            "channel into data/channel/<UTC timestamp>/ (raw JSON + CSVs + manifest.json)."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/channel"), help="base output folder (default: data/channel)"
    )
    parser.add_argument(
        "--since", type=_iso_date, help="Analytics start date YYYY-MM-DD (default: channel creation date)"
    )
    parser.add_argument("--skip-comments", action="store_true", help="don't download comments")
    parser.add_argument(
        "--max-videos",
        type=_positive_int,
        metavar="N",
        help="only export the N most recent uploads (quick test runs)",
    )
    parser.add_argument(
        "--monetary",
        action="store_true",
        help="also request revenue/ad metrics (adds the yt-analytics-monetary.readonly scope; YPP only)",
    )
    parser.add_argument(
        "--create-reporting-jobs",
        action="store_true",
        help="create missing YouTube Reporting API jobs (reach/impressions, traffic, devices, ...)",
    )
    parser.add_argument(
        "--download-reports",
        action="store_true",
        help="download new Reporting API files into <out>/reporting/<report type>/ (skips existing)",
    )
    parser.add_argument(
        "--reporting-only",
        action="store_true",
        help="only run --create-reporting-jobs/--download-reports, skip the Data/Analytics export",
    )
    parser.add_argument(
        "--client-secret",
        type=Path,
        help="OAuth client JSON (default: $YT_CLIENT_SECRET or secrets/client_secret.json)",
    )
    parser.add_argument(
        "--token", type=Path, default=DEFAULT_TOKEN, help=f"cached OAuth token (default: {DEFAULT_TOKEN})"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="print the sign-in URL instead of opening a browser"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def configure_logging(verbose: bool) -> None:
    """Progress log to stdout."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    if not verbose:
        for noisy in ("googleapiclient", "google_auth_httplib2", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse flags, authorize, run the export. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reporting_only and not (args.create_reporting_jobs or args.download_reports):
        parser.error("--reporting-only needs --create-reporting-jobs and/or --download-reports")
    configure_logging(args.verbose)

    try:
        creds = get_credentials(
            monetary=args.monetary,
            client_secret=args.client_secret,
            token_path=args.token,
            open_browser=not args.no_browser,
        )
    except AuthSetupError as err:
        log.error("%s", err)
        return 2
    youtube, analytics_service, reporting_service = build_services(creds)

    options = ExportOptions(
        out_base=args.out,
        since=args.since,
        skip_comments=args.skip_comments,
        max_videos=args.max_videos,
        monetary=args.monetary,
        create_reporting_jobs=args.create_reporting_jobs,
        download_reports=args.download_reports,
        reporting_only=args.reporting_only,
    )
    manifest = Exporter(youtube, analytics_service, reporting_service, options).run()
    log.info(
        "Finished in %.0fs: %s | Data API units used: %d | errors: %d (details in manifest.json)",
        manifest["elapsed_seconds"],
        manifest["output_dir"],
        manifest["quota"]["data_api_units_used"],
        manifest["error_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
