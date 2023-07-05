import torch as T
from torch.func import jacfwd, jacrev, vmap, grad


class GradientMaps:
    def __init__(self, grid_points, classifier_activations, inverter) -> None:
        self.inverter = inverter
        self.grid_points: T.Tensor = grid_points
        self.classifier_activations = classifier_activations

    @T.no_grad()
    def norm_jac_classif_wrt_inverted_grid(self):
        return vmap(
            lambda *a, **kw: T.linalg.matrix_norm(
                jacrev(self.classifier_activations)(*a, **kw)
            ),
            chunk_size=10_000,
        )(self.inverter(self.grid_points))

    @T.no_grad()
    def norm_jac_classif_and_inversion_wrt_grid(self):
        return vmap(
            lambda *a, **kw: T.linalg.matrix_norm(
                jacrev(
                    lambda points: self.classifier_activations(self.inverter(points))
                )(*a, **kw)
            ),
            chunk_size=5_000,
        )(self.grid_points)

    @T.no_grad()
    def norm_jac_classif_and_inversion_wrt_grid_for_class(self, class_: int):
        return vmap(
            lambda *a, **kw: T.linalg.vector_norm(
                grad(
                    lambda points: self.classifier_activations(self.inverter(points))[
                        class_
                    ]
                )(*a, **kw)
            ),
            chunk_size=5_000,
        )(self.grid_points)

    @T.no_grad()
    def norm_jac_classif_wrt_inverted_grid_for_class(self, class_: int):
        return vmap(
            lambda *a, **kw: T.linalg.vector_norm(
                grad(lambda points: self.classifier_activations(points)[class_])(
                    *a, **kw
                )
            ),
            chunk_size=10_000,
        )(self.inverter(self.grid_points))

    @T.no_grad()
    def norm_jac_inversion_wrt_grid(self):
        return vmap(
            lambda *a, **kw: T.linalg.matrix_norm(jacfwd(self.inverter)(*a, **kw)),
            chunk_size=2_500,
        )(self.grid_points)

    @T.no_grad()
    def smallest_sing_value_inversion_wrt_grid(self):
        def smallest_sing_value(*args, **kwargs):
            jac = jacfwd(self.inverter)(*args, **kwargs)
            return T.linalg.svdvals(jac)[1]

        return vmap(smallest_sing_value, chunk_size=10_000)(self.grid_points)

    @T.no_grad()
    def unprojection_grad_inversion_wrt_grid(self):
        # ASSUMING SQUARE GRID.
        grid_size = (self.grid_points[1] - self.grid_points[0])[0].item()

        with T.device(self.grid_points.device):
            dx_pos = self.grid_points + T.tensor([[grid_size, 0.0]])
            dx_neg = self.grid_points - T.tensor([[grid_size, 0.0]])

            dy_pos = self.grid_points + T.tensor([[0.0, grid_size]])
            dy_neg = self.grid_points - T.tensor([[0.0, grid_size]])

            D_x: T.Tensor = (self.inverter(dx_pos) - self.inverter(dx_neg)) / (
                2 * grid_size
            )
            D_y: T.Tensor = (self.inverter(dy_pos) - self.inverter(dy_neg)) / (
                2 * grid_size
            )

            D = T.sqrt(D_x.square().sum(dim=-1) + D_y.square().sum(dim=1))
        return D
