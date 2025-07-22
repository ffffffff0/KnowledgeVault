import importlib
import inspect

ChatModel = globals().get("ChatModel", {})
CvModel = globals().get("CvModel", {})
EmbeddingModel = globals().get("EmbeddingModel", {})
RerankModel = globals().get("RerankModel", {})
Seq2txtModel = globals().get("Seq2txtModel", {})
TTSModel = globals().get("TTSModel", {})

MODULE_MAPPING = {
    "chat_model": ChatModel,
    "cv_model": CvModel,
    "embedding_model": EmbeddingModel,
    "rerank_model": RerankModel,
    "sequence2txt_model": Seq2txtModel,
    "tts_model": TTSModel,
}

package_name = __name__

for module_name, mapping_dict in MODULE_MAPPING.items():
    full_module_name = f"{package_name}.{module_name}"
    module = importlib.import_module(full_module_name)

    base_class = None
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and name == "Base":
            base_class = obj
            break
    if base_class is None:
        continue

    for _, obj in inspect.getmembers(module):
        if (
            inspect.isclass(obj)
            and issubclass(obj, base_class)
            and obj is not base_class
            and hasattr(obj, "_FACTORY_NAME")
        ):
            if isinstance(obj._FACTORY_NAME, list):
                for factory_name in obj._FACTORY_NAME:
                    mapping_dict[factory_name] = obj
            else:
                mapping_dict[obj._FACTORY_NAME] = obj

__all__ = [
    "ChatModel",
    "CvModel",
    "EmbeddingModel",
    "RerankModel",
    "Seq2txtModel",
    "TTSModel",
]
