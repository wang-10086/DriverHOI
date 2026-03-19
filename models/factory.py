def create_model(config):
    model_type = getattr(config, 'MODEL_TYPE', 'DriverHOI')

    common_kwargs = dict(
        num_act=config.NUM_ACT,
        num_cat=config.NUM_CAT,
        node_dim=config.NODE_DIM,
        num_devices=config.NUM_DEVICES,
        ablation=config.ABLATION_MODE
    )

    if model_type == 'DriverHOI':
        from .driverhoi import DriverHOIModel
        model = DriverHOIModel(**common_kwargs)
        print(f"[Factory] 已创建模型: DriverHOI (GPNN, 本文提出)")

    elif model_type == 'TransHOI':
        from .transhoi import TransHOIModel
        model = TransHOIModel(**common_kwargs)
        print(f"[Factory] 已创建对比模型: TransHOI (Transformer交叉注意力)")

    elif model_type == 'SCG-HOI':
        from .scghoi import SCGHOIModel
        model = SCGHOIModel(**common_kwargs)
        print(f"[Factory] 已创建对比模型: SCG-HOI (空间条件图, 单次卷积)")

    else:
        raise ValueError(
            f"未知的 MODEL_TYPE: '{model_type}'\n"
            f"可选值: 'DriverHOI', 'MLP-HOI', 'TransHOI', 'SCG-HOI'"
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Factory] 总参数量: {total_params:,} | 可训练参数量: {trainable_params:,}")

    return model