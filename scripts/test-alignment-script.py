import argparse

from fibsem.structures import BeamType
from fibsem.tools.beam_shift_alignment_test import run_beam_shift_alignment_test


def main():
    """Main function to run the test alignment script."""

    parser = argparse.ArgumentParser(
        description="Run the test alignment script for FIBSEM microscope."
    )
    parser.add_argument(
        "--scan_rotation",
        type=int,
        default=180,
        help="Scan rotation in degrees (default: 180)",
    )
    parser.add_argument(
        "--hfw",
        type=float,
        default=150e-6,
        help="High Field Width (HFW) in meters (default: 150e-6)",
    )
    parser.add_argument(
        "--beam_type",
        type=str,
        choices=["ION", "ELECTRON"],
        default="ION",
        help="Type of beam to use (default: ION)",
    )

    args = parser.parse_args()

    scan_rotation = args.scan_rotation
    hfw = args.hfw
    beam_type = BeamType[args.beam_type]

    run_beam_shift_alignment_test(scan_rotation=scan_rotation, hfw=hfw, beam_type=beam_type)


if __name__ == "__main__":
    main()
