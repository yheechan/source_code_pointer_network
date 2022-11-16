import torch
import torch.nn.functional as F
import torch.utils.data as tud
from tqdm.auto import tqdm
import copy


def predictNoBeam(
    prefix,
    postfix,
    label,
    model=None,
    device=None
):


    # ********************* INSTANTIATE MODEL INPUT DATA *********************

    # [batch_size (1), token_length (64)]
    prefix = torch.tensor(prefix).unsqueeze(dim=0)
    postfix = torch.tensor(postfix).unsqueeze(dim=0)
    label = torch.tensor(label).unsqueeze(dim=0)

    prefix = prefix.to(device)
    postfix = postfix.to(device)
    label = label.to(device)




    # ********************* PREDICT TOKEN SEQUENCE *********************

    # [label_len (128 labels), batch_size, output_size (2 binary)]
    results = model(prefix, postfix, label)


    tok_list = []
    for i in range(results.shape[0]):

        # the token with highest probability
        preds = results[i].argmax(1).flatten()

        # append each token to list
        tok_list.append(preds.item())
    
    return tok_list