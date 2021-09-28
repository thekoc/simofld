from pathlib import Path
import json

for p in Path('.').glob("*user-num*"):
    with open(p) as f:
        j = json.load(f)
        for i in j:
            i['type'] = 'user_num'
    
    with open(p, 'w') as f:
        json.dump(j, f)
