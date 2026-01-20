#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import argparse
from plantbgc.output.KEGG_Proxy import run_kegg_proxy


class KEGGProxyCommand(object):
    """
    Command wrapper: plantbgc KEGG_Proxy ...
    """

    def add_subparser(self, subparsers):
        p = subparsers.add_parser(
            "KEGG_Proxy",
            help="Assign KEGG proxy labels (primary/secondary/mixed/review) for candidate loci using KO mappings.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        p.add_argument("--cds-tsv", required=True,
                       help="Gene/CDS-level TSV with locus assignment (e.g., nonzero.tsv).")
        p.add_argument("--ko-tsv", required=True,
                       help="KO mapping file (query_id -> KO). Typically from BlastKOALA/GhostKOALA.")
        p.add_argument("--out-prefix", required=True,
                       help="Output prefix (writes *_KEGGproxy*.tsv).")

        # Input column names
        p.add_argument("--locus-col", default="potentialBGC",
                       help="Locus id column in cds-tsv.")
        p.add_argument("--gene-col", default="sequence_id",
                       help="Gene/protein id column in cds-tsv to match KO file query id.")
        p.add_argument("--ko-col", default=None,
                       help="KO column name in ko-tsv if it has a header. If None, assume 2nd column is KO.")

        # KO sets
        p.add_argument("--ko-sec", default=None,
                       help="Optional: secondary-metabolism KO list file (one Kxxxxx per line).")
        p.add_argument("--ko-prim", default=None,
                       help="Optional: primary-metabolism KO list file (one Kxxxxx per line).")

        # Optional refinement (OFF by default to match the paper rule)
        p.add_argument("--primary-ratio-mode", action="store_true", default=False,
                       help="If enabled: require primary_ratio>=threshold to label a locus as primary when sec_score==0.")
        p.add_argument("--primary-ratio-thres", type=float, default=0.8,
                       help="Primary ratio threshold used when --primary-ratio-mode is enabled.")

        p.set_defaults(func=self)

    def run(self, cds_tsv, ko_tsv, out_prefix,
            locus_col="potentialBGC", gene_col="sequence_id", ko_col=None,
            ko_sec=None, ko_prim=None,
            primary_ratio_mode=False, primary_ratio_thres=0.8):
        run_kegg_proxy(
            cds_tsv=cds_tsv,
            ko_tsv=ko_tsv,
            out_prefix=out_prefix,
            locus_col=locus_col,
            gene_col=gene_col,
            ko_col=ko_col,
            ko_sec_file=ko_sec,
            ko_prim_file=ko_prim,
            primary_ratio_mode=primary_ratio_mode,
            primary_ratio_thres=primary_ratio_thres,
        )
