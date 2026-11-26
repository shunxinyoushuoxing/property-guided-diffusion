"""Persistent JSON-lines energy worker for separate predictor environments."""
import argparse
import contextlib
import json
import sys
from ase import Atoms
from .backends import make_predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON file, or read initialization JSON from stdin')
    args = parser.parse_args()
    if args.config:
        with open(args.config, encoding='utf-8-sig') as handle:
            config = json.load(handle)
    else:
        config = json.loads(sys.stdin.readline())
    with contextlib.redirect_stdout(sys.stderr):
        predictor = make_predictor(config)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            structures = [Atoms(numbers=a['numbers'], positions=a['positions'], cell=a['cell'], pbc=a['pbc'])
                          for a in request['structures']]
            with contextlib.redirect_stdout(sys.stderr):
                values = predictor.energies(structures).tolist()
            response = {'energies': values}
        except Exception as exc:
            response = {'error': str(exc)}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
