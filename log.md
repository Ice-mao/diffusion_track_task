# 修改track文件
## 🩹 Adding a Task

新建一个dataset:

`diffusion_policy/dataset/pusht_image_dataset.py`

新建一个env_runner:(不创建)
`diffusion_policy/env_runner/pusht_image_runner.py`

新建一个task：

`diffusion_policy/config/task/pusht_image.yaml`

Make sure that shape_meta correspond to input and output shapes for your task. Make sure env_runner._target_ and dataset._target_ point to the new classes you have added. When training, add task=<your_task_name> to train.py's arguments.

## 🩹 Adding a Method
新建一个workspace来组织整个流程

`diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`

新建一个policy或者复用：

`diffusion_policy/policy/diffusion_unet_image_policy.py`

新建一个总的配置文件：

`diffusion_policy/config/train_diffusion_unet_image_workspace.yaml`

Make sure your workspace yaml's _target_ points to the new workspace class you created.