from .activation import Clampazzo, Swish
from .edge import EdgeMap
from .inception import Inception
from .inhibition import LocalResponseInhibition
from .normalization import LRN
from .pooling import SegmentPooling
from .sparse import Sparse
from .temporal import CrossActivator

Clampz = Clampazzo
LRI = LocalResponseInhibition

FeaturePooling = SegmentPooling