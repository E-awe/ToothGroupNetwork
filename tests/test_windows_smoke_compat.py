"""CPU checks: import isolation, rotations, and unmodified DDIM mathematics.

Run: python -m unittest discover -s tests -p test_windows_smoke_compat.py -v
No CUDA extension is compiled or executed.
"""
import importlib.abc
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from utils.rotation6d import matrix_to_rotation_6d, rotation_6d_to_matrix

ROOT = Path(__file__).resolve().parents[1]
BASE = '92967a68191846ab38dcdb2ae7914e19ecd0d396'


class RotationTests(unittest.TestCase):
    def test_rows_batch_dtype_and_roundtrip(self):
        r = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
        d6 = matrix_to_rotation_6d(r)
        torch.testing.assert_close(d6, torch.tensor([0., -1., 0., 1., 0., 0.], dtype=r.dtype))
        torch.testing.assert_close(rotation_6d_to_matrix(d6), r)
        q, _ = torch.linalg.qr(torch.randn(2, 3, 3, 3, dtype=torch.float64))
        q[..., 2, :] *= torch.det(q)[..., None]
        result = rotation_6d_to_matrix(matrix_to_rotation_6d(q))
        self.assertEqual(result.shape, (2, 3, 3, 3))
        torch.testing.assert_close(result, q)
        torch.testing.assert_close(result @ result.transpose(-1, -2), torch.eye(3, dtype=q.dtype).expand_as(q))
        torch.testing.assert_close(torch.det(result), torch.ones(2, 3, dtype=q.dtype))

    def test_autograd(self):
        d6 = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(rotation_6d_to_matrix, (d6,)))
        r = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(matrix_to_rotation_6d, (r,)))


class StartupTests(unittest.TestCase):
    def test_main_without_pytorch3d_or_chamfer(self):
        code = r'''
import importlib.abc, sys
class NoExtensions(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('pytorch3d', 'chamfer', 'xformers'):
            raise ModuleNotFoundError('Intentionally unavailable: ' + fullname)
sys.meta_path.insert(0, NoExtensions())
import main
import runners.pretrain
from models.MeshMAE import Mesh_encoder, Mesh_mae
from utils.config import cfg_from_yaml_file
assert cfg_from_yaml_file('config/TADPM.yaml').model.NAME == 'TADPM'
try:
    Mesh_mae()
except RuntimeError as e:
    assert 'compiled chamfer extension' in str(e)
else:
    raise AssertionError('Missing chamfer must produce a clear Mesh_mae error')
assert not any(k.startswith('pytorch3d') for k in sys.modules)
print('Native main/train/test/pretrain imports passed without optional extensions')
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class DiffusionTests(unittest.TestCase):
    def test_ddim_matches_official_reference_and_state_keys(self):
        # Only the REFERENCE redirects CUDA transfers to CPU, in this test process.
        # The patched model executes natively on CPU; no network/condition is mocked.
        from models.diffusion import diffusion
        source = subprocess.check_output(
            ['git', 'show', BASE + ':models/diffusion.py'], cwd=ROOT, text=True
        )
        source = source.replace(".to('cuda')", ".to('cpu')")
        reference = types.ModuleType('official_diffusion_reference')
        torch.set_num_threads(2)
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor):
            exec(compile(source, 'official_diffusion_reference.py', 'exec'), reference.__dict__)
            official = reference.diffusion(1827).eval()
            actual = diffusion(1827).eval()
            actual.load_state_dict(official.state_dict(), strict=True)
            self.assertEqual(list(actual.state_dict()), list(official.state_dict()))
            torch.testing.assert_close(actual.betas, official.betas, rtol=0, atol=0)
            context = torch.randn(1, 32, 1827)
            torch.manual_seed(42)
            expected = official(None, context)
        torch.manual_seed(42)
        result = actual(None, context)
        self.assertEqual(result.shape, (1, 32, 9))
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertFalse(any(k in actual.state_dict() for k in actual._buffers))


if __name__ == '__main__':
    unittest.main()
