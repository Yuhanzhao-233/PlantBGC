"""Merge utilities for PlantBGC Predict outputs.

This module converts a scored, row-wise BGC list (<prefix>.bgc.tsv) into a
locus-level table by:

1) Merging consecutive rows with identical Pfam signatures.
2) Assigning a locus ID (potentialBGC) based on record order and an allowed gap.

It is intended to be called as a post-processing step after Predict writers have
finished writing outputs.
"""

import os
import pandas as pd


def write_sequence_order_file(sequence_ids, out_path, col="sequence_id", sep="\t"):
    """Write a simple record-order file.

    Parameters
    ----------
    sequence_ids : list[str]
        Record IDs in the exact order they were processed.
    out_path : str
        Target file path (often <output>/sequence_ids_output.csv).
    col : str
        Column name to write.
    sep : str
        Delimiter to use (default is tab to match existing helper scripts).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(col + "\n")
        for sid in sequence_ids:
            f.write(str(sid) + "\n")
    return out_path


def _merge_consecutive_same_pfam(bgc_df, seq_col="sequence_id", pfam_col="pfam_ids"):
    """Merge consecutive rows that share the same Pfam signature.

    This mirrors the behavior in your standalone script merge_bgc_prediction.py:
    if two adjacent rows have the same pfam_ids, concatenate their sequence_id
    values using commas.
    """
    if bgc_df is None or bgc_df.empty:
        return bgc_df

    merged_rows = []
    current_seq_ids = []
    current_row = None

    for _, row in bgc_df.iterrows():
        if current_row is None:
            current_row = row.copy()
            current_seq_ids = [str(row[seq_col])]
            continue

        if str(row.get(pfam_col, "")) == str(current_row.get(pfam_col, "")):
            current_seq_ids.append(str(row[seq_col]))
        else:
            current_row[seq_col] = ",".join(current_seq_ids)
            merged_rows.append(current_row)
            current_row = row.copy()
            current_seq_ids = [str(row[seq_col])]

    if current_row is not None:
        current_row[seq_col] = ",".join(current_seq_ids)
        merged_rows.append(current_row)

    return pd.DataFrame(merged_rows)


def _load_sequence_order(sequence_order_path, col="sequence_id"):
    """Load an order file and return the ordered sequence_id list.

    The function first tries TSV (tab) to match write_sequence_order_file().
    If the column is not found, it falls back to CSV (comma).
    """
    if not sequence_order_path or (not os.path.exists(sequence_order_path)):
        return None

    df = pd.read_csv(sequence_order_path, sep="\t")
    if col not in df.columns:
        df = pd.read_csv(sequence_order_path)
    if col not in df.columns:
        raise ValueError("sequence order file does not contain column: {}".format(col))

    return df[col].astype(str).tolist()


def _first_id(seq_group, first=True):
    """Return the first or last id from a comma-separated id group."""
    ids = str(seq_group).split(",")
    return ids[0] if first else ids[-1]


def assign_potential_bgc_ids(
    merged_df,
    ordered_sequence_ids,
    *,
    seq_col="sequence_id",
    out_col="potentialBGC",
    max_gap=0,
    min_consecutive=3,
    require_total_pfam_gt=None,
    pfam_col="pfam_ids",
):
    """Assign locus IDs (potentialBGC) based on order, gap, and chain length.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        Output from _merge_consecutive_same_pfam().
    ordered_sequence_ids : list[str]
        The global record order used as the reference axis.
    max_gap : int
        Allowed number of missing records between adjacent hits.
        Internally, we merge when (pos_next - pos_prev) <= (max_gap + 1).
    min_consecutive : int
        Minimum chain length to be considered a valid locus.
    require_total_pfam_gt : int | None
        Optional additional constraint similar to your pfamcheck2 variant:
        (pfam_count(prev) + pfam_count(next)) must be > this value.
    """
    df = merged_df.copy()
    df[out_col] = pd.NA

    if df.empty or not ordered_sequence_ids:
        return df

    pos = {sid: i for i, sid in enumerate(ordered_sequence_ids)}

    def pfam_count(small_id):
        if require_total_pfam_gt is None or small_id is None:
            return 0
        mask = df[seq_col].astype(str).str.split(",").str[-1] == str(small_id)
        vals = df.loc[mask, pfam_col].values
        if len(vals) == 0:
            return 0
        pfams = str(vals[0]).split(";") if str(vals[0]).strip() else []
        return len([p for p in pfams if p])

    # Collect hits (rows that can be located on the reference order axis).
    hits = []
    for ridx, row in df.iterrows():
        last_id = _first_id(row[seq_col], first=False)
        if last_id in pos:
            hits.append((pos[last_id], ridx, last_id))

    hits.sort(key=lambda x: x[0])

    bgc_index = 0
    chain = []
    prev_p = None

    def flush_chain(chain_idxs):
        nonlocal bgc_index
        if len(chain_idxs) >= min_consecutive:
            bgc_index += 1
            for rr in chain_idxs:
                df.at[rr, out_col] = bgc_index

    for p, ridx, sid in hits:
        if prev_p is None:
            chain = [ridx]
            prev_p = p
            continue

        if p - prev_p <= (max_gap + 1):
            if require_total_pfam_gt is not None:
                prev_sid = ordered_sequence_ids[prev_p] if prev_p < len(ordered_sequence_ids) else None
                if (pfam_count(prev_sid) + pfam_count(sid)) <= require_total_pfam_gt:
                    flush_chain(chain)
                    chain = [ridx]
                    prev_p = p
                    continue

            chain.append(ridx)
            prev_p = p
        else:
            flush_chain(chain)
            chain = [ridx]
            prev_p = p

    flush_chain(chain)

    # Re-factorize to ensure IDs are 1..K without gaps.
    if df[out_col].notna().any():
        df[out_col] = pd.factorize(df[out_col])[0] + 1

    # Move the ID column to the front to match prior scripts.
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index(out_col)))
    return df[cols]


def postprocess_merge_outputs(
    *,
    output_dir,
    output_prefix,
    sequence_order_path=None,
    max_gap=0,
    min_consecutive=3,
    require_total_pfam_gt=None,
):
    """Run the merge pipeline inside a Predict output directory.

    Steps
    -----
    1) Read <output_dir>/<output_prefix>.bgc.tsv
    2) Merge consecutive identical-Pfam rows -> merged_bgc_output_with_pfam.tsv
    3) Assign potentialBGC IDs -> merged_bgc_output_mergedBGC_with_all_rows(_gap).tsv
    """
    bgc_tsv = os.path.join(output_dir, output_prefix + ".bgc.tsv")
    if not os.path.exists(bgc_tsv):
        raise FileNotFoundError("Missing BGC TSV: {}".format(bgc_tsv))

    bgc_df = pd.read_csv(bgc_tsv, sep="\t")

    # Step 1: merge consecutive rows by Pfam signature.
    merged_with_pfam = _merge_consecutive_same_pfam(bgc_df)
    out1 = os.path.join(output_dir, "merged_bgc_output_with_pfam.tsv")
    merged_with_pfam.to_csv(out1, sep="\t", index=False)

    # Step 2: assign locus IDs.
    ordered = _load_sequence_order(sequence_order_path) if sequence_order_path else None
    if ordered is None:
        # Fallback: derive a best-effort order from the TSV itself.
        ordered = [_first_id(s, first=False) for s in merged_with_pfam["sequence_id"].astype(str).tolist()]

    labeled = assign_potential_bgc_ids(
        merged_with_pfam,
        ordered_sequence_ids=ordered,
        max_gap=max_gap,
        min_consecutive=min_consecutive,
        require_total_pfam_gt=require_total_pfam_gt,
    )

    out2_name = "merged_bgc_output_mergedBGC_with_all_rows_gap.tsv" if max_gap > 0 else "merged_bgc_output_mergedBGC_with_all_rows.tsv"
    out2 = os.path.join(output_dir, out2_name)
    labeled.to_csv(out2, sep="\t", index=False)

    return {"merged_with_pfam": out1, "merged_loci": out2}
