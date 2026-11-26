"""Small synthetic checks; no datasets or pretrained weights are required."""
import unittest
import numpy as np
import torch
from ase import Atoms
from pgd.geometry import make_batch, map_to_template
from pgd.models.diffusion import CrystalDiffusion
from pgd.models.discrete import D3PM
from pgd.search import Member, Search, selection_probabilities, resample
from pgd.predictors.property import PropertyEvaluator


def architecture():
    return dict(time_dim=8, discrete_steps=4, transition='uniform', steps=8,
                beta={'scheduler_mode': 'cosine'}, sigma={'sigma_begin': 0.01, 'sigma_end': 0.1},
                decoder=dict(hidden_dim=16, num_layers=2, num_freqs=4, ln=True))


class MethodTests(unittest.TestCase):
    def test_discrete_kernel(self):
        for transition in ['uniform', 'absorbing']:
            kernel = D3PM(4, 5, transition_type=transition)
            torch.testing.assert_close(kernel.q_mats.sum(-1), torch.ones(5, 4))
            stationary = torch.full((4,4), 0.25) if transition == 'uniform' else torch.zeros(4,4)
            if transition == 'absorbing':
                stationary[:,0] = 1
            torch.testing.assert_close(kernel.q_mats[-1], stationary)

    def test_fitness_sharing(self):
        def member(numbers):
            return Member(None, None, torch.tensor(numbers), None, 1.0, -1.0)
        population=[member([1,1]),member([1,1]),member([2,2])]
        np.testing.assert_allclose(selection_probabilities(population,2,.15), [.25,.25,.5])
        np.testing.assert_allclose(selection_probabilities(population,2), [1/3]*3)
        population[-1].reward=10
        selected=resample(population,np.random.default_rng(0),elites=1)
        self.assertEqual(selected[0].reward,10)
        self.assertIsNot(selected[0],population[-1])

    def test_sampling_and_forward_noise(self):
        model=CrystalDiffusion(architecture(),[0,1,2]).eval()
        model.requires_grad_(False)
        batch=make_batch({'num_atoms':4})
        torch.manual_seed(11)
        out,_=model.sample(batch,time_start=3)
        self.assertEqual(out['frac_coords'].shape,(4,3))
        self.assertTrue(torch.isfinite(out['lattices']).all())
        self.assertTrue(((out['frac_coords'] >= 0) & (out['frac_coords'] < 1)).all())
        self.assertFalse(out['frac_coords'].requires_grad)
        from pgd.search import noise_state
        parent=Member(out['frac_coords'],torch.zeros(1,6),torch.tensor([1,1,2,2]),None,0,0)
        noise=noise_state(model,batch,parent,2)
        self.assertEqual(noise['init_types'].shape,(4,))

    def test_symmetry_constraints(self):
        with self.assertRaises(ValueError):
            make_batch({'num_atoms':2,'spacegroup':2})
        model=CrystalDiffusion(architecture(),[0,1,2],
            [{'axis':2,'lower':0,'upper':1,'allowed_indices':[1]}])
        mask=model.vocabulary.position_mask(torch.rand(4,3))
        self.assertEqual(mask[:,1].sum(),4)
        self.assertFalse(mask[:,[0,2]].any())

    def test_template_mapping(self):
        template=Atoms(numbers=[1,1,1],positions=[[0,0,0],[2,0,0],[4,0,0]],cell=[8,8,8],pbc=True)
        candidate=Atoms(numbers=[2,2],positions=[[4.05,0,0],[.05,0,0]],cell=[8,8,8],pbc=True)
        mapped=map_to_template(candidate,template,[dict(atomic_numbers=[2],template_indices=[0,1,2],allow_vacancies=True)],.2)
        self.assertEqual(mapped.numbers.tolist(),[2,2])
        np.testing.assert_allclose(mapped.positions,[[0,0,0],[4,0,0]])

    def test_energy_recipe(self):
        class Predictor:
            def energies(self, states):
                return np.array([0.,3.,1.])
        atoms=Atoms(numbers=[1],positions=[[0,0,0]],cell=[8,8,8],pbc=True)
        recipe=dict(mode='path_barrier',states=[{}, {}, {}],site_indices=[0])
        self.assertEqual(PropertyEvaluator(Predictor(),recipe)(atoms),3)
        recipe.update(mode='linear_steps',coefficients=[[-1,1,0],[0,-1,1]],reference_value=1)
        self.assertEqual(PropertyEvaluator(Predictor(),recipe)(atoms),2)

    def test_search_greedy_and_best_tracking(self):
        class Generator:
            device='cpu'
            def __init__(self):
                self.vocabulary=type('V',(),{'numbers':(0,1)})()
                self.beta_scheduler=type('B',(),{'timesteps':8})()
            def eval(self):return self
            def requires_grad_(self,value):return self
            def sample(self,*args,**kwargs):return {},None
        class Engine(Search):
            def evaluate(self,output):
                self.evaluations += 1
                value=10-self.evaluations
                return Member(torch.zeros(1,3),torch.zeros(1,6),torch.ones(1,dtype=torch.long),None,value,-value)
        from unittest.mock import patch
        opts=dict(population=2,candidates=2,iterations=2,noise_start=3,resampling='none')
        engine=Engine(Generator(),None,None,opts)
        with patch('pgd.search.noise_state',return_value={}):
            population,best,history=engine.run()
        self.assertEqual(best.value,0)
        self.assertEqual(engine.evaluations,10)
        self.assertLessEqual(history[-1]['best_value'],history[0]['best_value'])


if __name__=='__main__':
    unittest.main()
