#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
)
import logging
import os
try:
    # If the plantbgc module is not accessible, we are running as a script so we need to add ".." to PATH.
    from plantbgc import __version__
except ImportError:
    import sys
    parent = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    print('DEV mode: Adding "{}" to PATH'.format(parent))
    sys.path.append(parent)
    from plantbgc import __version__

import argparse
import matplotlib
matplotlib.use('Agg')
from plantbgc.command.prepare import PrepareCommand
from plantbgc.command.download import DownloadCommand
from plantbgc.command.Predict import PredictCommand
from plantbgc.command.merge import MergeCommand
from plantbgc.command.train import TrainCommand
from plantbgc.command.info import InfoCommand
from plantbgc.command.GO_Proxy import GOProxyCommand
from plantbgc.command.KEGG_Proxy import KEGGProxyCommand

import sys

def _fix_subparsers(subparsers):
    if sys.version_info[0] == 3:
        subparsers.required = True
        subparsers.dest = 'cmd'


class plantbgcParser(argparse.ArgumentParser):

    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            message = argparse._('unrecognized arguments: %s') % ' '.join(argv)
            if hasattr(args, 'cmd') and args.cmd:
                # Show help for specific command, catch exit and print message
                try:
                    super(plantbgcParser, self).parse_args(args=[args.cmd, '--help'])
                except:
                    pass
            self.exit(2, "{}\n{}\n".format("="*80, message))
        return args

    def error(self, message):
        self.print_help()
        formatted_message = "{}\n{}\n".format("="*80, message) if message else None
        self.exit(2, formatted_message)


def run(argv=None):
    import warnings
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="numpy.core.umath_tests is an internal NumPy module")
    warnings.filterwarnings("ignore", message="np.asscalar(a) is deprecated")
    warnings.filterwarnings("ignore", message="`wait_time` is not used anymore")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # Disable TensorFlow debug messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = plantbgcParser(prog='plantbgc',
                           description='plantbgc - Biosynthetic Gene Cluster detection and classification',
                           formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging."
    )

    # Sub commands
    subparsers = parser.add_subparsers(
        title='Available Commands',
        metavar='COMMAND',
        dest='cmd',
        help='Use: plantbgc COMMAND --help for command-specific help')

    _fix_subparsers(subparsers)

    commands = [
        DownloadCommand(),
        PrepareCommand(),
        PredictCommand(),
        MergeCommand(),
        GOProxyCommand(),
        KEGGProxyCommand(),  # <- NEW
        TrainCommand(),
        InfoCommand()
    ]

    for command in commands:
        command.add_subparser(subparsers)

    args = parser.parse_args(argv)
    args_dict = {k: v for k, v in args.__dict__.items() if k not in ['cmd', 'func', 'debug']}

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)-7s %(asctime)s   %(message)s',
                        level=log_level, datefmt="%d/%m %H:%M:%S")


    args.func.run(**args_dict)


def main(argv=None):
    print( file=sys.stderr)
    try:
        run(argv)
    except KeyboardInterrupt:
        print(' Interrupted by the user')
        sys.exit(0)
    except Exception as e:
        message = e.args[0] if e.args else ''
        logging.exception(message)
        logging.error('='*80)
        logging.error('plantbgc failed with %s%s', type(e).__name__, ': {}'.format(message) if message else '')
        if len(e.args) > 1:
            logging.error('='*80)
            for arg in e.args[1:]:
                logging.error(arg)
        logging.error('='*80)
        exit(1)


if __name__ == '__main__':
    main(sys.argv[1:])
