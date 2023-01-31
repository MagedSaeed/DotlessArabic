import os
import subprocess

import requests


def download_file(file_url, file_name, output_dir):
    outfile = os.path.join(output_dir, file_name)
    response = requests.get(
        file_url,
        stream=True,
    )
    with open(outfile, "wb") as output:
        output.write(response.content)
    return True


def execute_bash(command, print_to_console=True):
    for line in command.splitlines():
        if line.strip():
            try:
                results = subprocess.check_output(line, shell=True)
                if print_to_console:
                    print(results)
            except Exception as e:
                print(e)
                raise e


def log_content(
    content,
    strip_lines=True,
    results_file=None,
    print_to_console=True,
):
    if not results_file and not print_to_console:
        print(
            "WARNING: log_results is called but with no results file and print_to_console is False"
        )
        return
    if results_file:
        with open(results_file, "a") as file:
            file.write("#" * 100)
            file.write("\n")
            for line in content.strip().splitlines():
                if strip_lines:
                    line = line.strip()
                file.write(line)
                file.write("\n")
            file.write("#" * 100)
            file.write("\n")
    if print_to_console:
        print("#" * 100)
        for line in content.strip().splitlines():
            if strip_lines:
                line = line.strip()
            print(line)
        print("#" * 100)
