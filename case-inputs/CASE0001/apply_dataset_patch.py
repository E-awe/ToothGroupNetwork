"""Apply only the packaged before-only Dataset patch; keep a unique source backup."""
import ast
import uuid
from pathlib import Path


def apply(repo):
    target=Path(repo)/'datasets/DentalDataset.py'
    source=target.read_text(encoding='utf-8')
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DentalDataset')
    if any(isinstance(n,ast.FunctionDef) and n.name=='_getitem_before_only' for n in cls.body):
        print('Dataset before-only mode already present; no changes applied.')
        return None
    lines=(Path(__file__).parent/'patches/dental-before-only.patch').read_text(encoding='utf-8').splitlines(keepends=True)
    hunks=[];old=[];new=[];active=False
    for line in lines:
        if line.startswith('@@'):
            if active: hunks.append((''.join(old),''.join(new)))
            old=[];new=[];active=True
        elif active and line[:1] in [' ','+','-']:
            if line[0] in [' ','-']: old.append(line[1:])
            if line[0] in [' ','+']: new.append(line[1:])
    if active: hunks.append((''.join(old),''.join(new)))
    updated=source
    for before,after in hunks:
        if updated.count(before)!=1:
            raise RuntimeError('Dataset differs from the reviewed patch context; no file changed. Compare patches/dental-before-only.patch with your Dataset.')
        updated=updated.replace(before,after,1)
    ast.parse(updated)
    backup=Path(repo)/'.case_input_backups'/f'DentalDataset_{uuid.uuid4().hex[:8]}.py'
    backup.parent.mkdir(parents=True,exist_ok=True)
    backup.write_bytes(target.read_bytes())
    target.write_text(updated,encoding='utf-8')
    print(f'Applied Dataset-only before-only extension; original backup: {backup}')
    return backup
