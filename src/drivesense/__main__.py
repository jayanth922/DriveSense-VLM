"""Entry point for the drivesense CLI (python -m drivesense)."""

from __future__ import annotations


def main() -> None:
    """Print package info. Subcommands implemented in later phases."""
    import drivesense

    print(f"DriveSense-VLM v{drivesense.__version__}")
    print("Run with --help for available subcommands (Phase 2a+).")


if __name__ == "__main__":
    main()
