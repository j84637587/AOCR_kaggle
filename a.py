import numpy as np
import matplotlib.pyplot as plt

import torch
from torchio import Compose, RandomAffine

sample_data = torch.rand(1, 232, 176, 50)

# Create the initial 3D plot

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.voxels(sample_data[0], edgecolor="k")

# Apply the Compose transformation
transform = Compose(
    [
        RandomAffine(
            degrees=(2),
            translation=(2 / 512, 2 / 512),
            scales=(0.001),
        )
    ]
)

transformed_data = transform(sample_data)

# Create the transformed 3D plot
fig_transformed = plt.figure()
ax_transformed = fig_transformed.add_subplot(111, projection="3d")
ax_transformed.voxels(transformed_data[0], edgecolor="k")

# Show the plots
plt.show()
