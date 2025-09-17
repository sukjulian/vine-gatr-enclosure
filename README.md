# ViNE-GATr enclosure

This implementation is a work in progress. This is *NOT* an official and supported implementation of ViNE-GATr.


## Quick start

Setup environment.

Experiments are run with the command
```bash
python scripts/$EXPERIMENT_ID.py training.num_epochs=$NUM_EPOCHS
```
where `EXPERIMENT_ID` is one of
```
arterial_surface
arterial_volume
mindboggle_101
random_dummy
shapenet_car
sphere
vine_mindboggle_101
```
and `NUM_EPOCHS` is the number of epochs to train the model. Model and optimiser checkpoints are
saved to the `checkpoints` directory.


## Command line interface
We use `hydra` to set a collection of experiment-specific parameters. Configurations for each
experiment can be found in the `config` folder. The following parameters are used across all
experiments:
```
id (str, optional): Identifier for the experiment used for writing checkpoints and logging.
training:
  automatic_mixed_precision (bool): Whether to train under automatic mixed precision. Observed to
    introduce instability (NaN) from time to time. Default: False
  batch_size (int): Batch size. Default: 1
  grad_accumulation_num_steps (int): Number of steps for gradient accumulation, effectively
    multiplying the batch size. Default: 1
  grad_clipping (bool): Whether to clip gradient norm to 1. Default: False
  lr (float): Learning rate. Default: 3e-4
  lr_decay_factor (float): Exponential factor for epoch-wise learning rate decay. Default: 1.0
  num_epochs (int): Number of training epochs.
  var_lr (float): Variant learning rate for separate parameter group. Parameters must be
    flagged in the model implementation. Default: lr
wandb (bool): Whether to log training statistics to Weights & Biases. Default: False
```
Model parameters, except for the one selecting the architecture, are passed to the respective model
constructor as is. All ViNE-GATr parameters are listed in its
[docstring](gatr_enclosure/models/vine_gatr.py).

### Arterial surface experiment
```
model:
  id (str): Identifier of the model architecture ("vine_gatr" or "gatr"). Default: "gatr"
  virtual_nodes_use_orientation (bool): Whether to preprocess the data to include surface normal and
    geodesic distance in the singular value decomposition.
num_samples (int, optional): Number of samples for subsampling during both training and inference.
training:
  loss (str): Training loss ("mse" or "l1"). Default: "l1"
  loss_term_factors:
    attractive (float): Attractive loss term factor.
    generic (float): Generic loss term factor.
```

### Arterial volume experiment
```
scaling:
  compression (float, optional): Compression factor for LaB-GATr farthest point sampling (0 to 1).
  num_samples (int, optional): Number of samples for subsampling during both training and inference.
  strategy (str): Scaling strategy ("gatr_and_subsampling", "lab_gatr" or "vine_gatr").
training:
  loss (str): Training loss ("mse" or "l1"). Default: "l1"
```

### Mindboggle-101
```
metric (str): Evaluation metric ("dice_coefficients" or "classification_accuracy"). Default:
  "classification_accuracy"
scaling:
  compression (float, optional): Compression factor for LaB-GATr farthest point sampling (0 to 1).
  num_samples (int, optional): Number of samples for subsampling during both training and inference.
  strategy (str): Scaling strategy ("gatr_and_subsampling", "lab_gatr", "vine_gatr" or "rng_gatr").
```

### Random dummy
```
model:
  id (str): Identifier of the model architecture ("gatr" or "vine_gatr").
num_pos (int): Number of points in the randomly generated dummy data.
training:
  loss_term_factors:
    generic (float): Generic loss term factor.
```

### ShapeNet-Car
```
model:
  compression (float, optional): Compression factor for LaB-GATr farthest point sampling (0 to 1).
  id (str): Identifier of the model architecture ("vine_gatr", "lab_gatr" or "rng_gatr").
  num_virtual_nodes (int): Number of ViNE-GATr virtual nodes and samples for RNG-GATr subsampling.
training:
  loss (str): Training loss ("mse" or "l1"). Default: "l1"
```

### Sphere
```
variance_gaussians (float): Variance of the mixture of Gaussians on the sphere.
```

### Virtual-node-embed Mindboggle-101
```
training:
  loss (str): Training loss ("attractive_repulsive", "attractive_manifold_repulsive" or
    "optimal_transport").
  loss_term_factors:
    attractive (float, optional): Attractive loss term factor.
    repulsive (float, optional): Repulsive loss term factor.
```

## Python installation
Requirements were exported to `requirements.txt` 

## License

ViNE-GATr enclosure is licensed under the [BSD-3-clause License](https://spdx.org/licenses/BSD-3-Clause.html). See [LICENSE](LICENSE) for the full license text.

