from __future__ import absolute_import, division, print_function

import logging

from plantbgc.command.base import BaseCommand
from plantbgc.output import merge as merge_utils


class MergeCommand(BaseCommand):
    command = 'merge'

    help = """\
Post-process Predict outputs by merging scored BGC regions into candidate loci.

This command does not run detection. It operates only on Predict outputs
(`*.bgc.tsv`) and produces merged TSV files.

Inputs
------
You can provide either:
  1) A Predict output directory, or
  2) A direct path to <prefix>.bgc.tsv

Outputs
-------
By default, outputs are written to the same directory as the input TSV (or the
Predict output directory). Two TSV files are created:
  * merged_bgc_output_with_pfam.tsv
  * merged_bgc_output_mergedBGC_with_all_rows.tsv
    (or *_gap.tsv when --merge-max-protein-gap > 0)

Examples
--------
  # Merge using a Predict output directory
  plantbgc merge path/to/out_dir --merge-max-protein-gap 2 --min-proteins 3

  # Merge using an explicit BGC TSV file
  plantbgc merge path/to/prefix.bgc.tsv --merge-max-protein-gap 2 --min-proteins 3

  # Provide an explicit prefix when the directory contains multiple *.bgc.tsv
  plantbgc merge path/to/out_dir --prefix prefix_name
"""

    def add_arguments(self, parser):
        parser.add_argument(
            'target',
            help='Predict output directory or a direct path to <prefix>.bgc.tsv',
        )
        parser.add_argument(
            '-o', '--output',
            required=False,
            help='Output directory for merged TSV files (defaults to input directory)',
        )
        parser.add_argument(
            '--prefix',
            dest='output_prefix',
            required=False,
            help='Prefix of the Predict outputs (used to locate <prefix>.bgc.tsv)',
        )
        parser.add_argument(
            '--sequence-order',
            dest='sequence_order',
            required=False,
            help='Optional path to sequence_ids_output.csv (column: sequence_id).',
        )

        group = parser.add_argument_group('Merge options', '')
        group.add_argument(
            '--merge-max-protein-gap',
            default=0,
            type=int,
            help='Allow up to N intervening proteins between hits (default: %(default)s)',
        )
        group.add_argument(
            '--min-proteins',
            default=3,
            type=int,
            help='Minimum number of connected hits to form a locus (default: %(default)s)',
        )
        group.add_argument(
            '--require-total-pfam-gt',
            default=None,
            type=int,
            help='Optional Pfam-count filter for linking adjacent hits (default: disabled).',
        )

    def run(
        self,
        target,
        output,
        output_prefix,
        sequence_order,
        merge_max_protein_gap,
        min_proteins,
        require_total_pfam_gt,
    ):
        resolved = merge_utils.resolve_bgc_tsv_and_output_dir(
            target_path=target,
            output_dir=output,
            output_prefix=output_prefix,
        )

        bgc_tsv = resolved['bgc_tsv']
        out_dir = resolved['output_dir']

        logging.info('Merging BGC regions from: %s', bgc_tsv)
        logging.info('Writing merged outputs to: %s', out_dir)

        result = merge_utils.postprocess_merge_from_bgc_tsv(
            bgc_tsv_path=bgc_tsv,
            output_dir=out_dir,
            sequence_order_path=sequence_order,
            max_gap=merge_max_protein_gap,
            min_consecutive=min_proteins,
            require_total_pfam_gt=require_total_pfam_gt,
        )

        logging.info('Created: %s', result['merged_with_pfam'])
        logging.info('Created: %s', result['merged_loci'])
