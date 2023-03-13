import paddle
import paddle.optimizer as optim


def get_gt_encoder(model, train_dataloaders, model_path=None,epoch_num=20):
    paddle.seed(0)
    device = paddle.get_device()
    if model_path is not None:
        if "gpu" in device:
            model.load_state_dict(paddle.load(model_path))
            model.cuda()
        else:
            model.load_state_dict(paddle.load(model_path, map_location="cpu"))
        print("gt encoder restored from the saved weights ...")
        return model
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-08,
                           weight_decay=0.0)
    # for epoch in range(epoch_num):
