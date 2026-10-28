


model_persister = None

class ModelPersister:
    def __init__(self):
        pass

    @classmethod
    def get_model_persister(cls):
        global model_persister
        if model_persister is not None:
            return model_persister


        from sys import platform
        is_on_mac_os = False

        if platform == "darwin":
            is_on_mac_os = True


        if is_on_mac_os:
            from .mlx_model_persister import MlxModelPersister
            model_persister = MlxModelPersister()
        else:
            from .safetensor_model_persister import SafetensorModelPersister
            model_persister = SafetensorModelPersister()
        return model_persister

    @classmethod
    def use(cls, kind):
        """Select the shard format for this process: 'safetensors' or 'mlx'.

        The PyTorch streaming path (VyanLLMBaseModel) needs safetensors shards on every platform,
        while the macOS MLX backend needs MLX shards. Each model class selects the format it needs
        when it is constructed, so the platform default no longer breaks the PyTorch path on Macs.
        """
        global model_persister
        if kind == 'safetensors':
            from .safetensor_model_persister import SafetensorModelPersister
            if not isinstance(model_persister, SafetensorModelPersister):
                model_persister = SafetensorModelPersister()
        elif kind == 'mlx':
            from .mlx_model_persister import MlxModelPersister
            if not isinstance(model_persister, MlxModelPersister):
                model_persister = MlxModelPersister()
        else:
            raise ValueError(f"unknown shard format {kind!r}")
        return model_persister

    def model_persist_exist(self, layer_name, saving_path):
        pass

    def persist_model(self, state_dict, layer_name, path):
        pass

    def load_model(self, layer_name, path):
        pass