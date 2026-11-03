"""pkg_resources yerine ince uyumluluk katmani (setuptools 84 pkg_resources'i kaldirdi; torchmetrics 0.10 onu ister).
Yeni paket indirmeden yalniz kullanilan iki ad saglanir. Yalniz COMET puanlamasinda sys.path'e eklenir."""
from importlib import metadata


class DistributionNotFound(Exception):
    pass


class _Dist:
    def __init__(self, name: str) -> None:
        try:
            self.version = metadata.version(name)
        except metadata.PackageNotFoundError as e:
            raise DistributionNotFound(name) from e
        self.project_name = name


def get_distribution(name: str) -> _Dist:
    return _Dist(name)
