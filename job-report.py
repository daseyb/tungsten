import subprocess
import json
import os
from pathlib import Path

USER_ID = "f003hxy"
DEFAULT_ARRAY_JOB_COUNT = 500

if __name__ == '__main__':
    result = subprocess.run(['squeue', '--user', USER_ID, "--json"], stdout=subprocess.PIPE)
    running_jobs_info = json.loads(result.stdout)

    overall_progess = {}
    running_tasks = {}
    waiting_tasks = {}
    total_tasks = {}
    job_names = {}

    for job in running_jobs_info["jobs"]:
        if job["job_state"] == "RUNNING":
            if job["array_job_id"]["set"] and job["array_task_id"]["set"]:
                job_id = job["array_job_id"]["number"]
                task_id = job["array_task_id"]["number"]

                if job_id not in job_names:
                    job_names[job_id] = job["name"]
                if job_id not in overall_progess:
                    overall_progess[job_id] = 0
                if job_id not in running_tasks:
                    running_tasks[job_id] = 0

                # Change report file depending on where your jobs output it to
                report_file = f"./report/output.{job_id}.{task_id}.out"

                if not Path(report_file).exists():
                    report_file = f"./report/output-render.{job_id}.{task_id}.out"

                if not Path(report_file).exists():
                    report_file = f"./report/output-normalgen.{job_id}.{task_id}.out"

                # Get progress from printout in report file. Adapt to your application
                try:
                    with open(report_file, 'r') as f:
                        last_line = f.readlines()[-1]
                    if last_line.startswith("Finished"):
                        progress = 1
                    elif last_line.startswith("Completed"):
                        comp, outof = last_line.split(" ")[1].split("/")
                        comp, outof = int(comp), int(outof)
                        progress = comp / outof
                    else:
                        progress = 0
                except:
                    progress = 0
                
                overall_progess[job_id] += progress
                running_tasks[job_id] += 1
        else:
            if job["array_task_string"] != "":
                job_id = job["array_job_id"]["number"]
                if job_id not in job_names:
                    job_names[job_id] = job["name"]
                
                running, total = map(int,job["array_task_string"].split("-"))
                total_tasks[job_id] = total
                waiting_tasks[job_id] = total - running
    
    if len(job_names) == 1:
        common_pref = ""
    else:
        common_pref = os.path.commonprefix(list(job_names.values()))

    max_len = max(map(len, job_names.values())) - len(common_pref) + 1

    for job in job_names.keys():
        if job not in waiting_tasks:
            waiting_tasks[job] = 0
        if job not in running_tasks:
            running_tasks[job] = 0
        if job not in total_tasks:
            total_tasks[job] = DEFAULT_ARRAY_JOB_COUNT
        if job not in overall_progess:
            overall_progess[job] = 0

        completed_tasks = total_tasks[job] - (running_tasks[job] + waiting_tasks[job])
        overall_progess[job] += completed_tasks
        overall_progess[job] /= total_tasks[job]

        print(f"{'...' if len(common_pref) > 0 else ''}{job_names[job][len(common_pref):]}:".ljust(max_len + 3), "[{:<{}}] {:.2f}%".format("=" * int(20 * overall_progess[job]), 20, overall_progess[job] * 100))


