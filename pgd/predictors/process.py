"""Run a predictor in its own Python environment without shell commands."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np


class ProcessPredictor:
    def __init__(self, python, config):
        env = dict(os.environ)
        root = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
        command = [python, '-u', '-m', 'pgd.predictors.worker']
        if not isinstance(config, dict):
            command.extend(['--config', str(Path(config).resolve())])
        self.process = subprocess.Popen(command,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, env=env)
        if isinstance(config, dict):
            self.process.stdin.write(json.dumps(config) + '\n')
            self.process.stdin.flush()

    def energies(self, structures):
        data = [{'numbers': a.numbers.tolist(), 'positions': a.positions.tolist(),
                 'cell': a.cell.tolist(), 'pbc': a.pbc.tolist()} for a in structures]
        self.process.stdin.write(json.dumps({'structures': data}) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Predictor worker exited; inspect its error output')
        response = json.loads(line)
        if 'error' in response:
            raise RuntimeError('Predictor worker: ' + response['error'])
        return np.asarray(response['energies'])

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
