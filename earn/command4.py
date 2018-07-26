import subprocess
subprocess.Popen('python InC_time.py --job_name="ps" --task_index=0',shell=True)
subprocess.Popen('python InC_time.py --job_name="worker" --task_index=0',shell=True)
subprocess.Popen('python InC_time.py --job_name="worker" --task_index=1',shell=True)

