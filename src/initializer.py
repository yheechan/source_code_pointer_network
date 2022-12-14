import seq2seq_model as sm
import torch.optim as optim
import torch.nn as nn

def initialize_model(
    learning_rate=0.001,
    weight_decay=0.0,
    embed_dim=100,
    hidden_size=200,
    n_layers=1,
    output_size=1,
    dropout=0.3,
    max_length=64,
    input_size=154,
    device=None,
    loss_fn_name='BCE'
):


    if loss_fn_name == 'BCE':
        loss_fn = nn.BCELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()


    model = sm.MySeq2Seq(
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        output_size=output_size,
        dropout=dropout,
        max_length=max_length,
        input_size=input_size,
        device=device
    )

    model.to(device)


    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )



    return model, loss_fn, optimizer