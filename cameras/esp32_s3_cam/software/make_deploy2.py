#!/usr/bin/env python3
"""Build a device-side deploy script: writes <local_file> to the board as
<target_name> (base64 transit), optionally hard-resets after.
Usage: make_deploy2.py <local_file> <target_name> [reset]"""
import base64
import sys

local, target = sys.argv[1], sys.argv[2]
do_reset = len(sys.argv) > 3 and sys.argv[3] == "reset"

with open(local, 'rb') as f:
    payload = base64.b64encode(f.read()).decode()

lines = [
    "import binascii, os",
    f'data = binascii.a2b_base64("{payload}")',
    f"with open('{target}', 'wb') as f:",
    "    f.write(data)",
    f"print('{target} written:', os.stat('{target}')[6], 'bytes')",
    "print('deploy-done')",
]
if do_reset:
    lines += [
        "import time, machine",
        "time.sleep(1)",
        "machine.reset()",
    ]

with open('deploy_out.py', 'w') as f:
    f.write("\n".join(lines) + "\n")
print("deploy_out.py built for", target, "reset=" + str(do_reset))
