"""Command line entry point for inference and population search."""
import argparse
import json
from pathlib import Path
import torch
from ase.io import read, write
from .geometry import make_batch, map_to_template, to_atoms
from .models.diffusion import CrystalDiffusion
from .search import Search


def read_json(path):
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description='Property guided diffusion inference')
    parser.add_argument('mode', choices=['sample', 'search'])
    parser.add_argument('--config', required=True, help='Inference-only JSON input')
    args = parser.parse_args()
    config = read_json(args.config)
    device = config.get('device', 'cpu')
    model = CrystalDiffusion.from_checkpoint(config['checkpoint'], read_json(config['model_metadata']), device)
    batch = make_batch(read_json(config['symmetry']), device)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(config.get('search', {}).get('seed', 0)))
    if args.mode == 'sample':
        for index in range(int(config.get('samples', 1))):
            result, _ = model.sample(batch, step_lr=config.get('step_lr', 1e-5))
            write(output/f'sample_{index:04d}.vasp', to_atoms(result), format='vasp', direct=True, sort=True)
        return
    from .predictors.backends import make_predictor
    from .predictors.property import PropertyEvaluator
    from .predictors.process import ProcessPredictor
    predictor_config = config['predictor_config']
    predictor = (ProcessPredictor(config['predictor_python'], predictor_config)
                 if config.get('predictor_python') else make_predictor(read_json(predictor_config)))
    try:
        evaluator = PropertyEvaluator(predictor, read_json(config['property_recipe']) if config.get('property_recipe') else None)
        postprocess = None
        if config.get('template_mapping'):
            mapping = read_json(config['template_mapping'])
            template = read(mapping['template'])
            postprocess = lambda atoms: map_to_template(atoms, template, mapping['groups'], mapping['max_distance'])
        initial_model = None
        if config.get('initial_checkpoint'):
            initial_model = CrystalDiffusion.from_checkpoint(config['initial_checkpoint'],
                read_json(config['initial_model_metadata']), device)
        engine = Search(model, batch, evaluator, config['search'], postprocess, initial_model)
        def callback(row, population, best):
            print(json.dumps(row), flush=True)
            with (output/'history.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, allow_nan=False)+'\n')
            write(output/'best.vasp', best.atoms, format='vasp', direct=True, sort=True)
        population, best, history = engine.run(callback)
        records = []
        for index, member in enumerate(population):
            name = f'final_{index:04d}.vasp'
            write(output/name, member.atoms, format='vasp', direct=True, sort=True)
            records.append(dict(file=name, value=member.value, reward=member.reward))
        (output/'results.json').write_text(json.dumps(dict(population=records, best_value=best.value,
            best_reward=best.reward, evaluations=engine.evaluations, failures=engine.failures), indent=2,
            allow_nan=False), encoding='utf-8')
    finally:
        if isinstance(predictor, ProcessPredictor):
            predictor.close()


if __name__ == '__main__':
    main()
