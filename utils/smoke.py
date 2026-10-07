"""Checkpoint-free forward validation; never a treatment prediction/evaluation."""
import json
import platform
from pathlib import Path
import torch


def configure_smoke_dataset(args, config):
    """Reuse existing case files, with deterministic test loading and batch size 1."""
    cfg = config.dataset.test
    root = Path(args.case_root or './case_0').resolve()
    fields = ('dataroot', 'paramroot', 'before_path', 'after_path', 'file')
    already_configured = all(Path(str(cfg[key])).exists() for key in fields)
    if args.case_root or root.is_dir():
        paths = {
            'dataroot': root / 'remesh_before',
            'paramroot': root / 'param_PLACEHOLDER',
            'before_path': root / 'pcd_before512',
            'after_path': root / 'pcd_after512_PLACEHOLDER',
            'file': root / 'case_ids.txt',
        }
        for key, fallback in [('paramroot', 'param'), ('after_path', 'pcd_after512')]:
            if not paths[key].exists():
                paths[key] = root / fallback
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError('Incomplete existing case_0 input: ' + ', '.join(missing))
        cfg.update({key: str(path) for key, path in paths.items()})
        cfg.npoint = 512
    elif not already_configured:
        raise FileNotFoundError(
            'Extract the EXISTING tadpm_official_upper_input.zip into this repository '
            '(creating ./case_0), or pass --case_root PATH. No data is regenerated.'
        )
    if cfg.NAME != 'DentalDataset':
        raise ValueError('Checkpoint-free case smoke test requires DentalDataset')
    cfg.train = False
    cfg.bs = 1


class SmokeRecorder:
    """Observe original forward calls and save finite raw slot outputs."""
    def __init__(self, model, args):
        self.args = args
        self.folder = Path(args.experiment_path)
        self.calls = {'Mesh_encoder': 0, 'Diffusion': 0, 'UViT': 0}
        self.steps = model.regressor.sampling_timesteps
        self.batches = []
        self.handles = []
        for name, module in [('Mesh_encoder', model.encoder),
                             ('Diffusion', model.regressor),
                             ('UViT', model.regressor.model)]:
            def hook(module, inputs, output, name=name):
                self.calls[name] += 1
            self.handles.append(module.register_forward_hook(hook))

    def save(self, index, masks, outputs):
        expected = (masks.shape[0], 32, 9)
        if tuple(outputs.shape) != expected:
            raise RuntimeError(f'Unexpected output shape {tuple(outputs.shape)}; expected {expected}')
        if not torch.isfinite(outputs).all():
            raise FloatingPointError('Non-finite TADPM/DDIM output; smoke test FAILED')
        ids = [int(i) for i in index]
        filename = f'smoke_batch_{len(self.batches):03d}.pt'
        torch.save({
            'case_ids': ids,
            'outputs': outputs.detach().cpu(),
            'masks': masks.detach().cpu(),
            'random_weights': True,
            'clinical_prediction': False,
        }, self.folder / filename)
        self.batches.append({
            'case_ids': ids, 'file': filename, 'shape': list(outputs.shape),
            'all_finite': True, 'masks': masks.detach().cpu().tolist(),
            'output_min': outputs.min().item(), 'output_max': outputs.max().item(),
        })

    def finish(self):
        for handle in self.handles:
            handle.remove()
        count = len(self.batches)
        expected = {'Mesh_encoder': 32 * count, 'Diffusion': count,
                    'UViT': self.steps * count}
        if not count or self.calls != expected:
            raise RuntimeError(f'Incomplete forward smoke test: calls={self.calls}, expected={expected}')
        cuda = self.args.device == 'cuda'
        report = {
            'status': 'SMOKE_OK', 'checkpoint': None,
            'encoder_checkpoint': self.args.encoder_ckpts or None,
            'random_weights': True, 'clinical_prediction': False,
            'after_placeholder_is_not_ground_truth': True,
            'add_metric_computed': False,
            'platform': platform.platform(), 'python': platform.python_version(),
            'torch': torch.__version__, 'torch_cuda_build': torch.version.cuda,
            'device': self.args.device,
            'gpu': torch.cuda.get_device_name() if cuda else None,
            'seed': self.args.seed, 'ddim_sampling_timesteps': self.steps,
            'forward_calls': self.calls, 'batches': self.batches,
        }
        (self.folder / 'smoke_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
        print(f'SMOKE_OK: DentalDataset -> TADPM -> Diffusion/UViT -> DDIM ({self.steps} steps)')
        print(f'Output/report directory: {self.folder}')
