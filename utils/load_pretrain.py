import torch
import os


def load_pretrain_weights(model, dataset_config, args, algo):
    if algo == "knowrare":
        pretrain_weights_path = os.path.join(dataset_config.root_dir, "pretrain_weights",
                                            'reconstruction', f"lstm.pth")
        state_dict = torch.load(pretrain_weights_path, weights_only=True, map_location=torch.device("cpu"))['model']
        model.load_embedor_state_dict(state_dict)
    else:
        print(f"No need to load pretrain weights for the algorithm {args.algo}")
    
    
    