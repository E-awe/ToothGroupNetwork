"""Dataset-mode boundary tests only; no old smoke test or network forward."""
import ast
import subprocess
import tempfile
import unittest
from pathlib import Path
from easydict import EasyDict
from datasets.DentalDataset import DentalDataset


class BeforeOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'mesh').mkdir();(self.root/'points').mkdir()
        (self.root/'ids.txt').write_text('1\n')
        self.cfg=EasyDict(before_only=True,train=False,npoint=512,
            dataroot=str(self.root/'mesh'),before_path=str(self.root/'points'),file=str(self.root/'ids.txt'))

    def tearDown(self):self.temp.cleanup()

    def test_absent_slots_require_no_after_or_rotation(self):
        sample=DentalDataset(self.cfg)[0]
        self.assertEqual(set(sample),{'index','feats','center','cordinates','faces','Fs','before_points','centroid','masks'})
        self.assertEqual(sample['before_points'].shape,(32,512,3))
        self.assertEqual(sample['masks'].sum(),0)

    def test_missing_partner_file_is_an_error(self):
        (self.root/'mesh/1_4.obj').write_text('')
        with self.assertRaises(FileNotFoundError):DentalDataset(self.cfg)[0]

    def test_training_and_after_targets_are_rejected(self):
        self.cfg.train=True
        with self.assertRaises(ValueError):DentalDataset(self.cfg)
        self.cfg.train=False
        for key in ['after_path','paramroot']:
            self.cfg[key]='must_not_be_read'
            with self.assertRaises(ValueError):DentalDataset(self.cfg)
            del self.cfg[key]

    def test_original_feature_and_paired_item_logic_unchanged(self):
        repo=Path(__file__).resolve().parents[1]
        before=ast.parse(subprocess.check_output(['git','show','3523d8e1db6a9136e3224165d68f5f1a6a478599:datasets/DentalDataset.py'],cwd=repo,text=True))
        after=ast.parse((repo/'datasets/DentalDataset.py').read_text())
        def get(tree,name):
            return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        self.assertEqual(ast.dump(get(before,'load_mesh_shape')),ast.dump(get(after,'load_mesh_shape')))
        # Only a leading opt-in dispatch was added to __getitem__.
        self.assertEqual([ast.dump(n) for n in get(before,'__getitem__').body],
                         [ast.dump(n) for n in get(after,'__getitem__').body[1:]])


if __name__=='__main__':unittest.main()
