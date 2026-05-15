# -*- coding: utf-8 -*-
import base64
import os

def restore():
    target = r'd:\PythonCodeStudy\WHT_LightChassisModel\wht_topo\run_topo.py'
    # The content will be appended via terminal commands in base64 chunks
    with open('chunks.txt', 'r') as f:
        full_b64 = f.read().replace('\n', '').replace(' ', '')
    
    decoded = base64.b64decode(full_b64).decode('utf-8')
    with open(target, 'w', encoding='utf-8') as f:
        f.write(decoded)
    print(f"Successfully restored {target}")

if __name__ == "__main__":
    restore()
