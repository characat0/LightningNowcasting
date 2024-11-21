from vastai import VastAI
import os
import re
import datetime
import time
import json
import sys

key = os.getenv('VAST_API_KEY')
assert key is not None, "missing vast api key"
repo = os.getenv('REPO_NAME')
assert repo is not None, "missing repository url"
subpackage = os.getenv('SUBPACKAGE')
assert subpackage is not None, "missing subpackage"
mlflow_password = os.getenv('MLFLOW_PASSWORD')

env = os.environ


sdk = VastAI(api_key=key)


def set_output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'{name}={value}', file=fh)

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
    if 'READY' in files:
        return 'READY'
    return 'UNKNOWN'

def launch_action_runner_with_gpu():
    instances_kwargs = {k.removeprefix('VAST_').lower():v for (k,v) in env.items() if k.startswith('VAST_') and k != 'VAST_API_KEY'}
    instance_env = ' '.join([f"-e {k.removeprefix('ENV_')}={v}" for (k,v) in env.items() if k.startswith('ENV_')])

    print(instances_kwargs)
    output: str = sdk.launch_instance(
        num_gpus="1", 
        onstart="./scripts/startup.sh",
        disk=64,
        env=instance_env,
        raw=True,
        ssh=True,
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
    set_output('instance_id', instance_id)

    start_time = datetime.datetime.now()
    try:
        while (status := check_status(instance_id)) in ['LOADING', 'UNKNOWN']:
            now = datetime.datetime.now()
            print(now, status)
            time.sleep(20)
            if (now - start_time).seconds > (60 * 10):
                print("Timeout while waiting instance to finish loading")
        print("Final status:", status)
        if status == 'ERROR':
            print(sdk.logs(INSTANCE_ID=instance_id))
            raise Exception("Error while loading instance")
        elif status == 'READY':
            print("Instance ready to receive jobs")
    except Exception as e:
        print("Destroying instance")
        sdk.destroy_instance(id=instance_id)
        raise e

def copy_data():
    instance_id = os.getenv('VAST_INSTANCE_ID')
    source = os.path.abspath(f"lib/{subpackage}/data")
    print(source)
    o = sdk.copy(src=source, dst=f"{instance_id}:/opt/data")


def start_training():
    instance_id = os.getenv('VAST_INSTANCE_ID')
    source = os.path.abspath(f"lib/{subpackage}/data/exp_raw")
    try:
        status = check_status(instance_id)
        if status == 'READY':
            r = repo.split('/')[1]
            print(o)
            
        while (status := check_status(instance_id)) in ['READY']:
            print(datetime.datetime.now(), status)
            time.sleep(60)
            print(sdk.logs(INSTANCE_ID=instance_id, tail='5'))
    finally:
        print("Destroying instance")
        sdk.destroy_instance(id=instance_id)
    print('='*64)
    print('='*64)
    print("Final status:", status)
    print(sdk.logs(INSTANCE_ID=instance_id))

if sys.argv[1] == 'LAUNCH_INSTANCE':
    launch_action_runner_with_gpu()
elif sys.argv[1] == 'START_TRAINING':
    start_training()