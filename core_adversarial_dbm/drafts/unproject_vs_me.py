import sys

sys.path.append("../core_adversarial_dbm")

import torch as T
import matplotlib.pyplot as plt
from run_experiments import read_and_prepare_data
from compute import gradient
from dotenv import load_dotenv

import gc

T.set_default_device("cuda" if T.cuda.is_available() else "cpu")

load_dotenv()

shape = (200, 200)

xx, yy = T.meshgrid(
    T.linspace(0.0, 1.0, shape[0]), T.linspace(0.0, 1.0, shape[1]), indexing="xy"
)
grid_points = T.stack([xx.ravel(), yy.ravel()], dim=1)

holder = read_and_prepare_data("mnist", "tsne", cache=False)
grad_maps = gradient.GradientMaps(
    grid_points, holder.classifier.activations, holder.nninv_model
)

unprojection = grad_maps.unprojection_grad_inversion_wrt_grid()

gc.collect()
T.cuda.memory.empty_cache()

my_method = grad_maps.norm_jac_inversion_wrt_grid()
gc.collect()
T.cuda.memory.empty_cache()

fig = plt.gcf()
ax1 = plt.subplot(121)
plt.title("UnProjection")
plt.imshow(
    unprojection.reshape(shape).cpu().numpy(),
    cmap="viridis",
    origin="lower",
    interpolation="none",
    extent=(0.0, 1.0, 0.0, 1.0),
)

plt.subplot(122)
plt.title("Mine")
plt.imshow(
    my_method.reshape(shape).detach().cpu().numpy(),
    cmap="viridis",
    origin="lower",
    interpolation="none",
    extent=(0.0, 1.0, 0.0, 1.0),
)
plt.show()
plt.close("all")

plt.imshow(
    unprojection.reshape(shape).cpu().numpy(),
    cmap="viridis",
    origin="lower",
    interpolation="none",
    extent=(0.0, 1.0, 0.0, 1.0),
)
plt.axis("off")
plt.gca().set_xlim(0.0, 1.0)
plt.gca().set_ylim(0.0, 1.0)
plt.savefig("UnProjection.png", bbox_inches="tight", pad_inches=0.0, dpi=200)
plt.close()

plt.imshow(
    my_method.reshape(shape).detach().cpu().numpy(),
    cmap="viridis",
    origin="lower",
    interpolation="none",
    extent=(0.0, 1.0, 0.0, 1.0),
)
plt.axis("off")
plt.gca().set_xlim(0.0, 1.0)
plt.gca().set_ylim(0.0, 1.0)
plt.savefig("MySpaceExpansion.png", bbox_inches="tight", pad_inches=0.0, dpi=200)
plt.close()
