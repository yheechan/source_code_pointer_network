from sklearn.model_selection import train_test_split
import torch
import torch, gc

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import timeit

import data
import data_loader as dl
import initializer as init
import trainer
import tester
# import predictor
import model_util as mu

import os


gc.collect()
torch.cuda.empty_cache()

# print(torch.cuda.memory_summary(device=None, abbreviated=False))

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")





proj_list =[
    'boringssl', 'c-ares',
    'freetype2', 'guetzli',
    'harfbuzz', 'libpng',
    'libssh', 'libxml2',
    'pcre', 'proj4',
    'sqlite3',
    'vorbis', 'woff2',
    'wpantund'
]





target_project = 0




for i in range(len(proj_list)):
    target_project = i
    print(proj_list[target_project]+'...\n')



    # get all data exept target project
    prefix_np, postfix_np,\
    label_np, label_prefix_np,\
    label_postfix_np, case_np = data.getTrainData(proj_list, proj_list[target_project])


    # get target project data
    test_prefix, test_postfix,\
    test_label, test_label_prefix,\
    test_label_postfix, test_case = data.getTestData(proj_list[target_project])





    train_prefix, val_prefix,\
    train_postfix, val_postfix,\
    train_label, val_label,\
    train_label_prefix, val_label_prefix,\
    train_label_postfix, val_label_postfix,\
    train_case, val_case = train_test_split(
        prefix_np, postfix_np,\
        label_np, label_prefix_np,\
        label_postfix_np, case_np,\
        test_size = 0.2, random_state = 43
    )




    train_dataloader, val_dataloader, test_dataloader =\
        dl.data_loader(
            train_prefix, val_prefix,
            train_postfix, val_postfix,
            train_label, val_label,
            train_label_prefix, val_label_prefix,
            train_label_postfix, val_label_postfix,
            train_case, val_case,


            test_prefix, test_postfix,
            test_label, test_label_prefix,
            test_label_postfix, test_case,

            batch_size=1000
        )





    # ====================
    # set parameters here
    # ====================

    overall_title = 'fc3'
    title = proj_list[target_project]
    epochs = 20 

    # max_len, source_code_tokens, token_choices = data.getInfo()

    learning_rate = 0.001
    weight_decay = 0.0

    embed_dim = 100 # 100
    hidden_size = 200 # 200
    n_layers = 1
    output_size = 1 # max(token_choices) + 1
    dropout = 0.0
    max_length = 64 # max_len
    input_size = 214 # max(token_choices) + 1
    device = device

    model_name = "seq2seq"
    optim_name = "Adam"
    loss_fn_name = "BCE"

    teacher_forcing_ratio = 0.75
    threshold = torch.tensor([0.5]).to(device)





    trainer.set_seed(42)

    model, loss_fn, optimizer = init.initialize_model(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        output_size=output_size,
        dropout=dropout,
        max_length=max_length,
        input_size=input_size,
        device=device,
        loss_fn_name=loss_fn_name
    )

    # print(model)





    model = mu.getModel(overall_title, 'fc3_boringssl')
    print(model)





    loss, acc, TT_acc = tester.test(
        test_dataloader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
        device=device,
        fn=overall_title,
        proj_nm=title,
        threshold=threshold
    )





    with open('../stat/'+overall_title, 'a') as f:
            # text = title + '\t |\tloss: ' + str(loss) + '\t |\tacc: ' + str(acc) + '\t |\t time: ' + str(round(end_time, 3)) + ' min\t |\t TT acc: ' + str(TT_acc)
            text = title + '\t |\tloss: ' + str(loss) + '\t |\tacc: ' + str(acc) + '\t |\t time: ' + str(round(0.0, 3)) + ' min\t |\t TT acc: ' + str(TT_acc) + '\n'
            f.write(text)