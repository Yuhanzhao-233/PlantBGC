from __future__ import (
    print_function,
    division,
    absolute_import,
)

import logging

import plantbgc.util
from plantbgc.command.base import BaseCommand
import os
from plantbgc import util
from Bio import SeqIO

from plantbgc.output.bgc_genbank import BGCGenbankWriter
from plantbgc.output.evaluation.pr_plot import PrecisionRecallPlotWriter
from plantbgc.output.evaluation.roc_plot import ROCPlotWriter
from plantbgc.output.readme import ReadmeWriter
from plantbgc.Predict.annotator import plantbgcAnnotator
from plantbgc.Predict.detector import plantbgcDetector
from plantbgc.Predict.classifier import plantbgcClassifier
from plantbgc.output.genbank import GenbankWriter
from plantbgc.output.evaluation.bgc_region_plot import BGCRegionPlotWriter
from plantbgc.output.cluster_tsv import ClusterTSVWriter
from plantbgc.output.evaluation.pfam_score_plot import PfamScorePlotWriter
from plantbgc.output.pfam_tsv import PfamTSVWriter
from plantbgc.output.antismash_json import AntismashJSONWriter


class PredictCommand(BaseCommand):
    command = 'Predict'

    help = """Run plantbgc Predict: Preparation, BGC detection, BGC classification and generate the report directory.

Examples:

  # Show detailed help 
  plantbgc Predict --help 

  # Detect BGCs in a nucleotide FASTA sequence using plantbgc model 
  plantbgc Predict sequence.fa


  # Add additional clusters detected using plantbgc model with a strict score threshold
  plantbgc Predict --continue --output sequence/ --label plantbgc_90_score --score 0.9 sequence/sequence.full.gbk
  """

    LOG_FILENAME = 'LOG.txt'
    PLOT_DIRNAME = 'evaluation'
    TMP_DIRNAME = 'tmp'

    def add_arguments(self, parser):

        parser.add_argument(dest='inputs', nargs='+', help="Input sequence file path (FASTA, GenBank, Pfam CSV)")

        parser.add_argument('-o', '--output', required=False, help="Custom output directory path")
        parser.add_argument('--limit-to-record', action='append',
                            help="Process only specific record ID. Can be provided multiple times")
        parser.add_argument('--minimal-output', dest='is_minimal_output', action='store_true', default=False,
                            help="Produce minimal output with just the GenBank sequence file")
        parser.add_argument('--prodigal-meta-mode', action='store_true', default=False,
                            help="Run Prodigal in '-p meta' mode to enable detecting genes in short contigs")
        parser.add_argument('--protein', action='store_true', default=False,
                            help="Accept amino-acid protein sequences as input (experimental). Will treat each file as a single record with multiple proteins.")

        group = parser.add_argument_group('BGC detection options', '')
        no_models_message = 'run "plantbgc download" to download models'
        detector_names = util.get_available_models('detector')
        group.add_argument('-d', '--detector', dest='detectors', action='append', default=[],
                           help="Trained detection model name ({}) or path to trained model pickle file. "
                                "Can be provided multiple times (-d first -d second)".format(
                               ', '.join(detector_names) or no_models_message))
        group.add_argument('--no-detector', action='store_true', help="Disable BGC detection")
        group.add_argument('-l', '--label', dest='labels', action='append', default=[],
                           help="Label for detected clusters (equal to --detector by default). "
                                "If multiple detectors are provided, a label should be provided for each one")
        group.add_argument('-s', '--score', default=0, type=float,
                           help="Average protein-wise plantbgc score threshold for extracting BGC regions from Pfam sequences (default: %(default)s)")
        group.add_argument('--merge-max-protein-gap', default=0, type=int,
                           help="Merge detected BGCs within given number of proteins (default: %(default)s)")
        group.add_argument('--merge-max-nucl-gap', default=0, type=int,
                           help="Merge detected BGCs within given number of nucleotides (default: %(default)s)")
        group.add_argument('--min-nucl', default=1, type=int,
                           help="Minimum BGC nucleotide length (default: %(default)s)")
        group.add_argument('--min-proteins', default=1, type=int,
                           help="Minimum number of proteins in a BGC (default: %(default)s)")
        group.add_argument('--min-domains', default=1, type=int,
                           help="Minimum number of protein domains in a BGC (default: %(default)s)")
        group.add_argument('--min-bio-domains', default=0, type=int,
                           help="Minimum number of known biosynthetic (as defined by antiSMASH) protein domains in a BGC (default: %(default)s)")

        group = parser.add_argument_group('BGC classification options', '')
        classifier_names = util.get_available_models('classifier')
        group.add_argument('-c', '--classifier', dest='classifiers', action='append', default=[],
                           help="Trained classification model name ({}) or path to trained model pickle file. "
                                "Can be provided multiple times (-c first -c second)".format(
                               ', '.join(classifier_names) or no_models_message))
        group.add_argument('--no-classifier', action='store_true', help="Disable BGC classification")
        group.add_argument('--classifier-score', default=0.5, type=float,
                           help="plantbgc classification score threshold for assigning classes to BGCs (default: %(default)s)")

    def run(self, inputs, output, detectors, no_detector, labels, classifiers, no_classifier,
            is_minimal_output, limit_to_record, score, classifier_score, merge_max_protein_gap, merge_max_nucl_gap,
            min_nucl,
            min_proteins, min_domains, min_bio_domains, prodigal_meta_mode, protein):
        if not detectors:
            detectors = ['plantbgc']
        if not classifiers:
            classifiers = ['class_classifier', 'activity_classifier']
        if not output:
            # if not specified, set output path to name of first input file without extension
            output, _ = os.path.splitext(os.path.basename(os.path.normpath(inputs[0])))

        if not os.path.exists(output):
            os.mkdir(output)

        # Save log to LOG.txt file
        logger = logging.getLogger('')
        logger.addHandler(logging.FileHandler(os.path.join(output, self.LOG_FILENAME)))

        # Define report dir paths
        tmp_path = os.path.join(output, self.TMP_DIRNAME)
        evaluation_path = os.path.join(output, self.PLOT_DIRNAME)
        output_file_name = os.path.basename(os.path.normpath(output))

        steps = []
        steps.append(plantbgcAnnotator(tmp_dir_path=tmp_path, prodigal_meta_mode=prodigal_meta_mode))
        if not no_detector:
            if not labels:
                labels = [None] * len(detectors)
            elif len(labels) != len(detectors):
                raise ValueError('A separate label should be provided for each of the detectors: {}'.format(detectors))

            for detector, label in zip(detectors, labels):
                steps.append(plantbgcDetector(
                    detector=detector,
                    label=label,
                    score_threshold=score,
                    merge_max_protein_gap=merge_max_protein_gap,
                    merge_max_nucl_gap=merge_max_nucl_gap,
                    min_nucl=min_nucl,
                    min_proteins=min_proteins,
                    min_domains=min_domains,
                    min_bio_domains=min_bio_domains
                ))

        writers = []
        writers.append(GenbankWriter(out_path=os.path.join(output, output_file_name + '.full.gbk')))
        writers.append(AntismashJSONWriter(out_path=os.path.join(output, output_file_name + '.antismash.json')))
        is_evaluation = False
        if not is_minimal_output:
            writers.append(BGCGenbankWriter(out_path=os.path.join(output, output_file_name + '.bgc.gbk')))
            writers.append(ClusterTSVWriter(out_path=os.path.join(output, output_file_name + '.bgc.tsv')))
            writers.append(PfamTSVWriter(out_path=os.path.join(output, output_file_name + '.pfam.tsv')))

            is_evaluation = True
            writers.append(PfamScorePlotWriter(out_path=os.path.join(evaluation_path, output_file_name + '.score.png')))
            writers.append(BGCRegionPlotWriter(out_path=os.path.join(evaluation_path, output_file_name + '.bgc.png')))
            writers.append(ROCPlotWriter(out_path=os.path.join(evaluation_path, output_file_name + '.roc.png')))
            writers.append(
                PrecisionRecallPlotWriter(out_path=os.path.join(evaluation_path, output_file_name + '.pr.png')))

        writers.append(ReadmeWriter(out_path=os.path.join(output, 'README.txt'), root_path=output, writers=writers))

        if not no_classifier:
            for classifier in classifiers:
                steps.append(plantbgcClassifier(classifier=classifier, score_threshold=classifier_score))

        # Create temp and evaluation dir
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        if is_evaluation:
            if not os.path.exists(evaluation_path):
                os.mkdir(evaluation_path)

        record_idx = 0
        for i, input_path in enumerate(inputs):
            logging.info('Processing input file %s/%s: %s', i + 1, len(inputs), input_path)
            with util.SequenceParser(input_path, protein=protein) as parser:
                for record in parser.parse():
                    if limit_to_record and record.id not in limit_to_record:
                        logging.debug('Skipping record %s not matching filter %s', record.id, limit_to_record)
                        continue

                    record_idx += 1
                    logging.info('=' * 80)
                    logging.info('Processing record #%s: %s', record_idx, record.id)
                    for step in steps:
                        step.run(record)

                    logging.info('Saving processed record %s', record.id)
                    for writer in writers:
                        writer.write(record)

        logging.info('=' * 80)
        for step in steps:
            step.print_summary()

        for writer in writers:
            writer.close()

        logging.info('=' * 80)
        logging.info('Saved plantbgc result to: {}'.format(output))
