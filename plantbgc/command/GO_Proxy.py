#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command: plantbgc goproxy

Parallel to Predict/Merge, this command runs GO mapping and GO-proxy labeling.

All comments are in English as requested.
"""

from __future__ import absolute_import, division, print_function

import os
import argparse

from plantbgc.output.GO_Proxy import run_go_proxy, pkg_data_path


class GOProxyCommand(object):

    command = "GO_Proxy"
    help = "GO proxy labeling (sec/prim/mixed/review) based on pfam2go and fixed rules."

    def add_subparser(self, subparsers):
        p = subparsers.add_parser(
            self.command,
            help=self.help,
            formatter_class=argparse.RawTextHelpFormatter
        )

        p.add_argument(
            "--cds-tsv",
            required=True,
            help="Gene/CDS-level TSV with locus assignment + Pfam list columns."
        )
        p.add_argument(
            "--out-prefix",
            required=True,
            help="Output prefix (writes *_GOproxy*.tsv)."
        )

        p.add_argument(
            "--pfam2go",
            default=pkg_data_path("pfam2go.txt"),
            help="Path to pfam2go.txt (default: package data/pfam2go.txt)."
        )

        # Optional: provide GO sets if not defined in plantbgc/features.py
        p.add_argument(
            "--go-sec",
            default=None,
            help="Optional file: secondary-metabolism GO IDs (one GO per line)."
        )
        p.add_argument(
            "--go-prim",
            default=None,
            help="Optional file: primary-metabolism GO IDs (one GO per line)."
        )

        # Column overrides
        p.add_argument("--locus-col", default="potentialBGC", help="Locus id column (default: potentialBGC).")
        p.add_argument("--pfam-col", default="pfams", help="Pfam list column (default: pfams).")
        p.add_argument("--gene-col", default="cds_id", help="Gene id column (default: cds_id).")

        p.set_defaults(func=self)

    def run(self, **kwargs):
        cds_tsv = kwargs["cds_tsv"]
        out_prefix = kwargs["out_prefix"]
        pfam2go_path = kwargs["pfam2go"]

        go_sec_file = kwargs.get("go_sec")
        go_prim_file = kwargs.get("go_prim")

        locus_col = kwargs.get("locus_col", "potentialBGC")
        pfam_col = kwargs.get("pfam_col", "pfams")
        gene_col = kwargs.get("gene_col", "cds_id")

        if not os.path.exists(cds_tsv):
            raise FileNotFoundError("CDS TSV not found: %s" % cds_tsv)
        if not os.path.exists(pfam2go_path):
            raise FileNotFoundError("pfam2go not found: %s" % pfam2go_path)

        locus_out, summary_out, gene_out = run_go_proxy(
            cds_tsv=cds_tsv,
            out_prefix=out_prefix,
            pfam2go_path=pfam2go_path,
            go_sec_file=go_sec_file,
            go_prim_file=go_prim_file,
            locus_col=locus_col,
            pfam_col=pfam_col,
            gene_col=gene_col,
        )

        print("GO proxy done.")
        print("Locus table :", locus_out)
        print("Summary     :", summary_out)
        print("Gene table  :", gene_out)
