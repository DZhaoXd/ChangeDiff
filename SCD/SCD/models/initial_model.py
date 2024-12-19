import numpy as np


def get_model_by_name(model_name, num_classes, in_width):
    if model_name == 'A2Net':
        from models.A2Net.A2Net import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)
    elif model_name == 'A2Net18':
        from models.A2Net_18.A2Net import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)
    elif model_name == 'BiSRNet':
        from models.BiSRNet.BiSRNet import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)
    elif model_name == 'SCanNet':
        from models.SCanNet.SCanNet import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes, input_size=in_width)
    elif model_name == 'SSCDL':
        from models.SSCDL.SSCDL import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)
    elif model_name == 'TED':
        from models.TED.TED import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)
    else:
        from models.A2Net.A2Net import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)


def adjust_learning_rate(args, optimizer, iter, max_batches, lr_factor=1):
    max_iter = max_batches * args.max_epochs
    warm_up_iter = np.floor(max_iter * 0.1)
    if args.lr_mode == 'poly':
        cur_iter = iter - warm_up_iter
        max_iter = max_iter - warm_up_iter
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if iter < warm_up_iter:
        lr = args.lr * 0.9 * (iter + 1) / warm_up_iter + 0.1 * args.lr  # warm_up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
