## algorithm generation
import os
from monai.apps.auto3dseg import BundleGen

work_dir = "./work_dir"
data_output_yaml = os.path.join(work_dir, "data_stats.yaml")
data_src_cfg = "./task.yaml"

bundle_generator = BundleGen(
    algo_path=work_dir,
    data_stats_filename=data_output_yaml,
    data_src_cfg_name=data_src_cfg,
)

bundle_generator.generate(work_dir, num_fold=5)