__all__ = ('RoIAlign', 'RoIPooling2D', 'RoIPooling2DTorch',)

try:
    from roi_align import RoIAlign
    from roi_pooling import RoIPooling2D, RoIPooling2DTorch
except ImportError:
    from .roi_align import RoIAlign
    from .roi_pooling import RoIPooling2D, RoIPooling2DTorch
