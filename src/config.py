import os

data_dir = "data"
base_dir = data_dir
input_dir = os.path.join(base_dir, "input")
output_dir = os.path.join(base_dir, "output")
tasks_dir = os.path.join(base_dir, "tasks")
tasks_done_dir = os.path.join(base_dir, "tasks/done")
tasks_error_dir = os.path.join(base_dir, "tasks/error")

seed = 123
device = "cuda"

instruct_pix2pix_batch_size = 2
