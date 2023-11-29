import subprocess
import argparse

for balanced in [0,1]:

    template = 'python train.py --balanced={} --simulation_name={}'
    repeat = 3
    simulation_names = [
        #"single-task",
        #"pcgrad",
        #"summed-loss",

        "project1-splitter",
        "project2-splitter",
        "project3-splitter",
        "normalized-project1-splitter",
        "normalized-project2-splitter",
        "normalized-project3-splitter",
        #"normalized-splitter",
    ]

    # Run commands in parallel
    processes = []

    for sim_name in simulation_names:
        for i in range(repeat):
            command = template.format(balanced, sim_name)
            process = subprocess.Popen(command, shell=True)
            processes.append(process)

    # Collect statuses
    output = [p.wait() for p in processes]