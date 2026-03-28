def __init__(self) -> None:
    self._registry: Dict[str, Type[AbstractMetric]] = {}
    self.register("euclidean", EuclideanMetric)
    self.register("gaussian_curved", GaussianCurvedMetric)
    self.register("minkowski", MinkowskiMetric)