import subprocess

template = 'python train.py --simulation_name={}'
repeat = 3
args = [
    "single-task",
    "pcgrad",
    "summed-loss",
    "normalized-splitter",
    "project-splitter",
    "normalized-project-splitter",
]

# Run commands in parallel
processes = []

for arg in args:
    for i in range(repeat):
        command = template.format(arg)
        process = subprocess.Popen(command, shell=True)
        processes.append(process)

# Collect statuses
output = [p.wait() for p in processes]