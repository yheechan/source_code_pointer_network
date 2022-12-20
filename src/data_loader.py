import torch
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

def data_loader(
    train_prefix, val_prefix, test_prefix,
    train_prefix_ids, val_prefix_ids, test_prefix_ids,

    train_postfix, val_postfix, test_postfix,
    train_postfix_ids, val_postfix_ids, test_postfix_ids,

    train_label, val_label, test_label,

    train_label_prefix, val_label_prefix, test_label_prefix,
    train_label_postfix, val_label_postfix, test_label_postfix,

    train_case, val_case, test_case,

    batch_size=1000
):
    
    train_prefix, val_prefix, test_prefix,\
    train_prefix_ids, val_prefix_ids, test_prefix_ids,\
    train_postfix, val_postfix, test_postfix,\
    train_postfix_ids, val_postfix_ids, test_postfix_ids,\
    train_label, val_label, test_label,\
    train_label_prefix, val_label_prefix, test_label_prefix,\
    train_label_postfix, val_label_postfix, test_label_postfix,\
    train_case, val_case, test_case,\
        = tuple(torch.tensor(data) for data in[
            train_prefix, val_prefix, test_prefix,
            train_prefix_ids, val_prefix_ids, test_prefix_ids,
            train_postfix, val_postfix, test_postfix,
            train_postfix_ids, val_postfix_ids, test_postfix_ids,
            train_label, val_label, test_label,
            train_label_prefix, val_label_prefix, test_label_prefix,
            train_label_postfix, val_label_postfix, test_label_postfix,
            train_case, val_case, test_case,
        ])
    
    # Create DataLoader for training data
    train_data = TensorDataset(
        train_prefix,
        train_prefix_ids,
        train_postfix,
        train_postfix_ids,
        train_label,
        train_label_prefix,
        train_label_postfix,
        train_case
    )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

    # Create DataLoader for validation data
    val_data = TensorDataset(
        val_prefix,
        val_prefix_ids,
        val_postfix,
        val_postfix_ids,
        val_label,
        val_label_prefix,
        val_label_postfix,
        val_case
    )
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, drop_last=True)

    # Create DataLoader for training data
    test_data = TensorDataset(
        test_prefix,
        test_prefix_ids,
        test_postfix,
        test_postfix_ids,
        test_label,
        test_label_prefix,
        test_label_postfix,
        test_case
    )
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader 