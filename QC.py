#!/usr/bin/env python3
"""
Filter Nanopore FASTQ reads by length and optional mean Q score.

Outputs:
  1. kept FASTQ
  2. optional failed FASTQ
  3. summary TXT
  4. summary JSON
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, TextIO, Tuple


FASTQ_SUFFIXES = (
    ".fastq.gz",
    ".fq.gz",
    ".fastq",
    ".fq",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QC cleanup for Nanopore FASTQ files. Reads are filtered by length, "
            "and optionally by mean Q score."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input FASTQ / FASTQ.GZ file.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Output file prefix. Default: ./<input_stem>.len<min>_<max>"
            "[.q<min_mean_q>]"
        ),
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=200,
        help="Minimum read length to keep. Default: 250",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=700,
        help="Maximum read length to keep. Default: 650",
    )
    parser.add_argument(
        "--min-mean-q",
        type=float,
        default=None,
        help="Optional minimum mean Phred Q score to keep a read.",
    )
    parser.add_argument(
        "--write-failed",
        action="store_true",
        help="Also write filtered-out reads to a failed FASTQ file.",
    )
    parser.add_argument(
        "--gzip-output",
        action="store_true",
        help="Write FASTQ outputs as gzip-compressed files.",
    )
    return parser.parse_args()


def strip_fastq_suffix(path: Path) -> str:
    name = path.name
    for suffix in FASTQ_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def build_output_prefix(
    input_path: Path,
    user_prefix: Optional[Path],
    min_length: int,
    max_length: int,
    min_mean_q: Optional[float],
) -> Path:
    if user_prefix is not None:
        return user_prefix

    name = f"{strip_fastq_suffix(input_path)}.len{min_length}_{max_length}"
    if min_mean_q is not None:
        q_token = str(min_mean_q).replace(".", "p")
        name = f"{name}.q{q_token}"
    return Path.cwd() / name


def open_maybe_gzip(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode)  # type: ignore[return-value]
    return path.open(mode)


def phred_mean(qual: str) -> float:
    if not qual:
        return 0.0
    return sum(ord(char) - 33 for char in qual) / len(qual)


def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction)


def describe_numeric(values: Iterable[float]) -> Dict[str, float]:
    collected = list(values)
    if not collected:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    ordered = sorted(collected)
    return {
        "count": float(len(ordered)),
        "mean": float(mean(ordered)),
        "min": float(ordered[0]),
        "p10": percentile(ordered, 0.10),
        "p25": percentile(ordered, 0.25),
        "p50": percentile(ordered, 0.50),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "max": float(ordered[-1]),
    }


def format_numeric_block(title: str, stats: Dict[str, float]) -> List[str]:
    return [
        title,
        f"  count : {int(stats['count'])}",
        f"  mean  : {stats['mean']:.2f}",
        f"  min   : {stats['min']:.2f}",
        f"  p10   : {stats['p10']:.2f}",
        f"  p25   : {stats['p25']:.2f}",
        f"  p50   : {stats['p50']:.2f}",
        f"  p75   : {stats['p75']:.2f}",
        f"  p90   : {stats['p90']:.2f}",
        f"  max   : {stats['max']:.2f}",
    ]


def record_failure_reasons(
    read_length: int,
    mean_q: float,
    min_length: int,
    max_length: int,
    min_mean_q: Optional[float],
) -> List[str]:
    reasons: List[str] = []
    if read_length < min_length:
        reasons.append("too_short")
    if read_length > max_length:
        reasons.append("too_long")
    if min_mean_q is not None and mean_q < min_mean_q:
        reasons.append("low_mean_q")
    return reasons


def validate_record(header: str, seq: str, plus: str, qual: str, read_index: int) -> None:
    if not header.startswith("@"):
        raise ValueError(f"Record {read_index}: header does not start with '@'")
    if not plus.startswith("+"):
        raise ValueError(f"Record {read_index}: plus line does not start with '+'")
    if len(seq) != len(qual):
        raise ValueError(
            f"Record {read_index}: sequence length ({len(seq)}) does not match "
            f"quality length ({len(qual)})"
        )


def fastq_ext(gzip_output: bool) -> str:
    return ".fastq.gz" if gzip_output else ".fastq"


def write_record(handle: TextIO, header: str, seq: str, plus: str, qual: str) -> None:
    handle.write(header)
    handle.write(seq)
    handle.write("\n")
    handle.write(plus)
    handle.write(qual)
    handle.write("\n")


def run_cleanup(args: argparse.Namespace) -> Tuple[Path, Optional[Path], Path, Path]:
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input FASTQ not found: {input_path}")
    if args.min_length < 0 or args.max_length < 0:
        raise ValueError("Length cutoffs must be non-negative")
    if args.min_length > args.max_length:
        raise ValueError("--min-length cannot be larger than --max-length")

    output_prefix = build_output_prefix(
        input_path=input_path,
        user_prefix=args.output_prefix.resolve() if args.output_prefix else None,
        min_length=args.min_length,
        max_length=args.max_length,
        min_mean_q=args.min_mean_q,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    kept_fastq = output_prefix.with_name(output_prefix.name + ".kept" + fastq_ext(args.gzip_output))
    failed_fastq = None
    if args.write_failed:
        failed_fastq = output_prefix.with_name(output_prefix.name + ".failed" + fastq_ext(args.gzip_output))
    summary_txt = output_prefix.with_name(output_prefix.name + ".summary.txt")
    summary_json = output_prefix.with_name(output_prefix.name + ".summary.json")

    total_reads = 0
    kept_reads = 0
    failed_reads = 0
    total_bases = 0
    kept_bases = 0
    failed_bases = 0

    all_lengths: List[int] = []
    kept_lengths: List[int] = []
    failed_lengths: List[int] = []
    all_qscores: List[float] = []
    kept_qscores: List[float] = []
    failed_qscores: List[float] = []

    fail_reason_counts: Counter[str] = Counter()
    fail_combo_counts: Counter[str] = Counter()

    with open_maybe_gzip(input_path, "rt") as in_handle, open_maybe_gzip(kept_fastq, "wt") as kept_handle:
        failed_handle: Optional[TextIO] = None
        try:
            if failed_fastq is not None:
                failed_handle = open_maybe_gzip(failed_fastq, "wt")

            while True:
                header = in_handle.readline()
                if not header:
                    break
                seq = in_handle.readline().rstrip("\n")
                plus = in_handle.readline()
                qual = in_handle.readline().rstrip("\n")

                total_reads += 1
                if plus == "" or qual == "":
                    raise ValueError(f"Input FASTQ ended mid-record at read {total_reads}")

                validate_record(header, seq, plus, qual, total_reads)

                read_length = len(seq)
                mean_q = phred_mean(qual)
                reasons = record_failure_reasons(
                    read_length=read_length,
                    mean_q=mean_q,
                    min_length=args.min_length,
                    max_length=args.max_length,
                    min_mean_q=args.min_mean_q,
                )

                total_bases += read_length
                all_lengths.append(read_length)
                all_qscores.append(mean_q)

                if reasons:
                    failed_reads += 1
                    failed_bases += read_length
                    failed_lengths.append(read_length)
                    failed_qscores.append(mean_q)
                    fail_reason_counts.update(reasons)
                    fail_combo_counts["+".join(reasons)] += 1
                    if failed_handle is not None:
                        write_record(failed_handle, header, seq, plus, qual)
                else:
                    kept_reads += 1
                    kept_bases += read_length
                    kept_lengths.append(read_length)
                    kept_qscores.append(mean_q)
                    write_record(kept_handle, header, seq, plus, qual)
        finally:
            if failed_handle is not None:
                failed_handle.close()

    kept_fraction = kept_reads / total_reads if total_reads else 0.0
    failed_fraction = failed_reads / total_reads if total_reads else 0.0

    summary = {
        "input_fastq": str(input_path),
        "filters": {
            "min_length": args.min_length,
            "max_length": args.max_length,
            "min_mean_q": args.min_mean_q,
        },
        "outputs": {
            "kept_fastq": str(kept_fastq),
            "failed_fastq": str(failed_fastq) if failed_fastq else None,
            "summary_txt": str(summary_txt),
            "summary_json": str(summary_json),
        },
        "counts": {
            "total_reads": total_reads,
            "kept_reads": kept_reads,
            "failed_reads": failed_reads,
            "kept_fraction": kept_fraction,
            "failed_fraction": failed_fraction,
            "total_bases": total_bases,
            "kept_bases": kept_bases,
            "failed_bases": failed_bases,
        },
        "failure_reason_counts": dict(sorted(fail_reason_counts.items())),
        "failure_reason_combinations": dict(sorted(fail_combo_counts.items())),
        "length_stats": {
            "all_reads": describe_numeric(all_lengths),
            "kept_reads": describe_numeric(kept_lengths),
            "failed_reads": describe_numeric(failed_lengths),
        },
        "mean_q_stats": {
            "all_reads": describe_numeric(all_qscores),
            "kept_reads": describe_numeric(kept_qscores),
            "failed_reads": describe_numeric(failed_qscores),
        },
    }

    text_lines = [
        "Nanopore FASTQ QC cleanup summary",
        "",
        f"Input FASTQ     : {input_path}",
        f"Kept FASTQ      : {kept_fastq}",
        f"Failed FASTQ    : {failed_fastq if failed_fastq else 'not written'}",
        f"Summary TXT     : {summary_txt}",
        f"Summary JSON    : {summary_json}",
        "",
        "Filters",
        f"  min_length : {args.min_length}",
        f"  max_length : {args.max_length}",
        f"  min_mean_q : {args.min_mean_q if args.min_mean_q is not None else 'disabled'}",
        "",
        "Read counts",
        f"  total_reads   : {total_reads}",
        f"  kept_reads    : {kept_reads}",
        f"  failed_reads  : {failed_reads}",
        f"  kept_fraction : {kept_fraction:.4f}",
        f"  failed_fraction: {failed_fraction:.4f}",
        "",
        "Base counts",
        f"  total_bases : {total_bases}",
        f"  kept_bases  : {kept_bases}",
        f"  failed_bases: {failed_bases}",
        "",
        "Failure reasons",
    ]
    if fail_reason_counts:
        for reason, count in sorted(fail_reason_counts.items()):
            text_lines.append(f"  {reason:12s}: {count}")
    else:
        text_lines.append("  none")

    text_lines.extend(["", "Failure reason combinations"])
    if fail_combo_counts:
        for combo, count in sorted(fail_combo_counts.items()):
            text_lines.append(f"  {combo:20s}: {count}")
    else:
        text_lines.append("  none")

    text_lines.extend([""])
    text_lines.extend(format_numeric_block("Length stats - all reads", summary["length_stats"]["all_reads"]))
    text_lines.extend([""])
    text_lines.extend(format_numeric_block("Length stats - kept reads", summary["length_stats"]["kept_reads"]))
    text_lines.extend([""])
    text_lines.extend(format_numeric_block("Length stats - failed reads", summary["length_stats"]["failed_reads"]))
    text_lines.extend([""])
    text_lines.extend(format_numeric_block("Mean Q stats - all reads", summary["mean_q_stats"]["all_reads"]))
    text_lines.extend([""])
    text_lines.extend(format_numeric_block("Mean Q stats - kept reads", summary["mean_q_stats"]["kept_reads"]))
    text_lines.extend([""])
    text_lines.extend(format_numeric_block("Mean Q stats - failed reads", summary["mean_q_stats"]["failed_reads"]))
    text_lines.append("")

    summary_txt.write_text("\n".join(text_lines))
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"Input FASTQ : {input_path}")
    print(f"Kept reads  : {kept_reads} / {total_reads} ({kept_fraction:.2%})")
    print(f"Kept FASTQ  : {kept_fastq}")
    if failed_fastq is not None:
        print(f"Failed FASTQ: {failed_fastq}")
    print(f"Summary TXT : {summary_txt}")
    print(f"Summary JSON: {summary_json}")

    return kept_fastq, failed_fastq, summary_txt, summary_json


def main() -> None:
    args = parse_args()
    run_cleanup(args)


if __name__ == "__main__":
    main()
