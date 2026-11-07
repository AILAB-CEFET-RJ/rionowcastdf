#!/usr/bin/env python3
"""Read-only Git housekeeping report; optional additive .gitignore update only.
Never deletes/moves files, unstages files, rewrites commits or changes running jobs.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess

PATTERNS='''
# Generated radar cache/policy artifacts (14j--14n)
/data/radar_intermediate_14j*/
/analysis_outputs_restored/**/*.sqlite
/analysis_outputs_restored/**/*.sqlite-*
/analysis_outputs_restored/**/*.parquet
/analysis_outputs_restored/**/paired_manifest.csv
/analysis_outputs_restored/**/policy_manifest.csv.gz
/analysis_outputs_restored/**/files.csv.gz
/analysis_outputs_restored/**/targets/
/analysis_outputs_restored/**/review_bundle.zip
/analysis_outputs_restored/**/smoke_tensors.npz
/analysis_outputs_restored/**/diagnostic_model_state.pt
/logs/
__pycache__/
*.pyc
*:Zone.Identifier
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',default='.')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--apply-gitignore',action='store_true')
    args=p.parse_args();root=Path(args.repo_root).resolve();out=Path(args.output_dir).resolve()
    if out.exists():p.error('Use a new report directory')
    actual=Path(subprocess.check_output(['git','-C',str(root),'rev-parse','--show-toplevel'],text=True).strip()).resolve()
    if actual!=root:p.error('repo-root must be the Git root')
    if args.apply_gitignore:
        path=root/'.gitignore'
        if path.is_symlink():p.error('Refusing symlink .gitignore')
        previous=path.read_text() if path.exists() else ''
        add=[line for line in PATTERNS.strip().splitlines() if line not in previous.splitlines()]
        if add:path.write_text(previous.rstrip()+'\n\n'+'\n'.join(add)+'\n')
    tracked=subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0')
    ignored=set(subprocess.check_output(['git','-C',str(root),'ls-files','-ci','--exclude-standard','-z']).decode().split('\0'))
    rows=[]
    for rel in tracked:
        if not rel:continue
        path=root/rel
        size=path.lstat().st_size if path.exists() or path.is_symlink() else 0
        if size>=50*1024*1024 or rel in ignored or ':Zone.Identifier' in rel:
            rows.append(dict(path=rel,size_bytes=size,tracked_but_ignored=rel in ignored,
                review_reason='tracked_ignored_or_large_or_zoneidentifier'))
    out.mkdir(parents=True)
    with (out/'tracked_artifacts_review.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','size_bytes','tracked_but_ignored','review_reason']);w.writeheader();w.writerows(rows)
    (out/'suggested_gitignore.txt').write_text(PATTERNS)
    (out/'git_status.txt').write_text(subprocess.check_output(['git','-C',str(root),'status','--short'],text=True))
    (out/'summary.json').write_text(json.dumps(dict(review_entries=len(rows),gitignore_updated=args.apply_gitignore,
        files_deleted=0,files_moved=0,git_index_modified=False,git_history_modified=False,
        note='Ignore rules do not remove tracked files or large blobs in previous commits. Review CSV before targeted git rm --cached; do not delete cache or active audit outputs.'),indent=2)+'\n')
    print((out/'summary.json').read_text())

if __name__=='__main__':main()
