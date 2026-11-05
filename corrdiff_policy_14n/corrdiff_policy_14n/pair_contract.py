"""Read helpers retained from 14k/14l; no inventory mutations."""
import hashlib
import json
CHANNELS = ['tcwv','t2m','u10','v10']+[f'{v}_{l}' for l in (850,500) for v in ('t','r','u','v')]

def dump(path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n')
    tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def choose(names, options, required=True):
    found=[x for x in options if x in names]
    if len(found)>1: raise ValueError('Ambiguous columns: '+str(found))
    if not found and required: raise ValueError('Missing columns: '+str(options))
    return found[0] if found else None

def qi(name): return '"'+name.replace('"','""')+'"'
def literal(s): return "'"+str(s).replace("'","''")+"'"

