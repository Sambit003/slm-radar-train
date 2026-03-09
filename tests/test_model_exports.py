from src.model import __all__ as package_exports
from src.model import (
    BaseMultiHeadClassifier,
    BertMultiHeadClassifier,
    GemmaMultiHeadClassifier,
    LlamaMultiHeadClassifier,
    focal_loss,
)
from src.model.modeling import __all__ as modeling_exports
from src.model.modeling import GemmaMultiHeadClassifier as ModelingGemma
from src.model.modeling import BaseMultiHeadClassifier as ModelingBase
from src.model.modeling import focal_loss as modeling_focal_loss


def test_model_package_exports_expected_public_api():
    assert package_exports == [
        "BaseMultiHeadClassifier",
        "focal_loss",
        "BertMultiHeadClassifier",
        "GemmaMultiHeadClassifier",
        "LlamaMultiHeadClassifier",
        "get_model_class",
        "register_model",
    ]
    assert BaseMultiHeadClassifier is not None
    assert BertMultiHeadClassifier is not None
    assert GemmaMultiHeadClassifier is not None
    assert LlamaMultiHeadClassifier is not None
    assert callable(focal_loss)


def test_modeling_module_reexports_gemma_surface():
    assert modeling_exports == [
        "BaseMultiHeadClassifier",
        "focal_loss",
        "GemmaMultiHeadClassifier",
    ]
    assert ModelingBase is BaseMultiHeadClassifier
    assert ModelingGemma is GemmaMultiHeadClassifier
    assert modeling_focal_loss is focal_loss
