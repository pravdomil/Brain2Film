import os

batch_size = 8
drive_dir = "/content/drive"
base_dir = os.path.join(drive_dir, "MyDrive/AI Cut Ultra")
input_dir = os.path.join(base_dir, "input")
output_dir = os.path.join(base_dir, "output")
tasks_dir = os.path.join(base_dir, "tasks")
tasks_done_dir = os.path.join(base_dir, "tasks/done")
tasks_error_dir = os.path.join(base_dir, "tasks/error")
