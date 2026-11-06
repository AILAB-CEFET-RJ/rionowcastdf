"""NumPy reader for reference tensors. Native CorrDiff adaptation remains separate."""
from pathlib import Path
import numpy as np
import zarr
from .contract_common import read,sha,digest_obj

class ReferenceDataset:
    def __init__(self,root,split='train',normalize_inputs=True):
        root=Path(root);summary=read(root/'summary.json')
        if not summary.get('complete'):raise ValueError('Incomplete dataset')
        for name in ('normalization.json','contract.json'):
            if sha(root/name)!=summary['artifact_sha256'][name]:raise ValueError('Dataset metadata changed')
        self.z=zarr.open_consolidated(str(root/(split+'.zarr')),mode='r')
        if not self.z.attrs.get('complete') or self.z.attrs['contract_sha256']!=digest_obj(read(root/'contract.json')):raise ValueError('Zarr contract mismatch')
        n=read(root/'normalization.json');self.mean=np.asarray(n['input_mean'],np.float32)[:,None,None]
        self.std=np.asarray(n['input_std'],np.float32)[:,None,None];self.normalize=normalize_inputs
    def __len__(self):return self.z['input'].shape[0]
    def __getitem__(self,i):
        x=self.z['input'][i];y=self.z['target'][i];mask=self.z['mask'][i]
        if self.normalize:x=(x-self.mean)/self.std
        return {'input':x,'target':y,'mask':mask,'timestamp_utc':int(self.z['timestamps'][i]),
                'source_timestamp_utc':int(self.z['source_timestamps'][i]),
                'patch_row':int(self.z['patch_row'][i]),'patch_col':int(self.z['patch_col'][i])}
