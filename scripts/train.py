from vastai import VastAI
import os
import re
import datetime
import time
import json

key = os.getenv('VAST_API_KEY')
assert key is not None, "missing vast api key"
repo = os.getenv('REPO_NAME')
assert repo is not None, "missing repository url"
subpackage = os.getenv('SUBPACKAGE')
assert subpackage is not None, "missing subpackage"
mlflow_password = os.getenv('MLFLOW_PASSWORD')

env = os.environ

instances_kwargs = {k.removeprefix('VAST_').lower():v for (k,v) in env.items() if k.startswith('VAST_') and k != 'VAST_API_KEY'}

sdk = VastAI(api_key=key)

print(instances_kwargs)

output: str = sdk.launch_instance(
    num_gpus="1", 
    onstart="./scripts/startup-train.sh",
    disk=64,
    env=f"-e REPO_NAME={repo} "
        f"-e SUBPACKAGE={subpackage} "
        f"-e MLFLOW_TRACKING_URI=http://mlflow.marcovela.com:6969/api "
        f"-e MLFLOW_TRACKING_USERNAME=lightning "
        f"-e MLFLOW_TRACKING_PASSWORD={mlflow_password}",
    raw=True,
    **instances_kwargs,
)
print(output)
match = re.search(r'\{.*?\}', output, re.DOTALL)
if not match:
    raise ValueError("No JSON object found in the input string.")
response: dict = json.loads(match.group(0))
if not response.get('success'):
    raise ValueError(f"Could not launch instance: {response.get('error')}")

instance_id = response['new_contract']

def check_status(instance_id):
    d = json.loads(sdk.show_instance(id=instance_id, raw=True))
    if d['intended_status'] == 'stopped':
        print(d['status_msg'])
        return "ERROR"
    if d['actual_status'] != 'running':
        return 'LOADING'
    files = sdk.execute(ID=instance_id, COMMAND="ls /root").splitlines()
    if 'FAILED' in files:
        return 'FAILED'
    if 'SUCCESS' in files:
        return 'FINISH'
    return 'WORKING'

start_time = datetime.datetime.now()
try:
    while (status := check_status(instance_id)) in ['LOADING', 'WORKING']:
        now = datetime.datetime.now()
        print(now, status)
        print(sdk.logs(INSTANCE_ID=instance_id, tail='5'))
        time.sleep(60)
    print('='*64)
    print('='*64)
    print("Final status:", status)
    print(sdk.logs(INSTANCE_ID=instance_id))
except Exception as e:
    print(e)
finally:
    print("Destroying instance")
    sdk.destroy_instance(id=instance_id)
