import torch
import numpy as np
from sklearn import metrics
import pandas as pd
import os
import json

def tensor2list(tensor):
    list = tensor.tolist()
    return list

def writeJson(
    prefix, postfix, label,
    label_prefix_binary, label_postfix_binary,
    prefix_likelihood, postfix_likelihood,
    proj_nm
):
    prefix, postfix, label,\
    label_prefix_binary, label_postfix_binary,\
    prefix_likelihood, postfix_likelihood = tuple(data.tolist() for data in [
        prefix, postfix, label,
        label_prefix_binary, label_postfix_binary,
        prefix_likelihood, postfix_likelihood
    ])

    json_data = {
        'prefix': prefix,
        'postfix': postfix,
        'label-type': label,
        'label-prefix': [label_prefix_binary],
        'label-postfix': [label_postfix_binary],
        'prefix-likelihood': [prefix_likelihood],
        'postfix-likelihood': [postfix_likelihood]
    }

    with open('../predicted/'+proj_nm, 'a') as f:
        info = json.dumps(json_data)
        f.write(info+'\n')



def test(
    test_dataloader=None,
    model=None,
    loss_fn=None,
    device=None,
    fn='test',
    proj_nm='cm',
    threshold=None
):
    

    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.to(device)


    # Put the model to evaluating mode
    model.eval()


    # Tracking variables
    fin_loss = []
    fin_acc = []
    fin_acc_TT = []

    # for confusion matrix
    tot_pred = torch.empty(0).to(device)
    tot_label = torch.empty(0).to(device)


    # For each batch in our validation set...
    for step, batch in enumerate(test_dataloader):

        loss = 0
        all_loss = 0

        # Load batch to GPU
        prefix, postfix, label,\
        label_prefix, label_postfix, case = tuple(t.to(device) for t in batch)

        # usage to save likelihoods of prefix and postfix as a whole
        # [batch_size, 64] --> [token len (128), batch_size, 1 (1 or 0)]
        total_labels = torch.cat((label_prefix, label_postfix), 1).permute(1, 0).unsqueeze(2).float()

        # Compute logits
        with torch.no_grad():

            # [token len (128), batch_size (1000), single likelihood (BCE - 1)]
            results = model(prefix, postfix, label)




            # ********** SPLIT PREDICTED LIKELIHOOD **********

            # torch.set_printoptions(sci_mode=False, precision=20)
            # [prefix prediction, postfix prediction] in likelihood
            prefix_likelihood, postfix_likelihood = torch.split(results, 64)

            # [batch_size (1000), token_len (64)]
            prefix_likelihood = prefix_likelihood.permute(1, 0, 2).squeeze()
            postfix_likelihood = postfix_likelihood.permute(1, 0, 2).squeeze()




            # add loss for each token predicted
            # [batch_size, label_length (128)]
            total_1000_128_pred_TT = torch.empty((0), dtype=torch.bool).to(device)

            for i in range(results.shape[0]):
                
                # ********** CALCULATE LOSS **********

                loss = loss_fn(results[i], total_labels[i])
                all_loss += loss

                # keep loss
                fin_loss.append(loss.item())



                # ********** EVALUATE PREDICTIONS **********

                # calculate accuracy
                # preds = results[i].argmax(1).flatten()
                # [batch_size, 1]
                preds = results[i]

                # say True with likelihood above threshold
                # [batch_size, 1]
                binary_preds = (preds>threshold)
    
                # say True that is predicted True
                # [batch_size, 1]
                guess_T = (binary_preds == True)

                # say True to what is really labeled True
                # [batch_size, 1]
                real_T = (total_labels[i] == 1)

                # say True when both prediction and label is True
                # [batch_size, 1]
                true_TT = torch.logical_and(guess_T, real_T)
                
                # print(true_TT)
                # eventually --> [batch_size, label_length (128)]
                total_1000_128_pred_TT = torch.cat((total_1000_128_pred_TT, true_TT), 1)




                # ********** LOG ACCURACY STATS **********
                
                # accuracy and loss calculation
                acc = (binary_preds == total_labels[i]).cpu().numpy().mean() * 100
                fin_acc.append(acc)

                tot_pred = torch.cat((tot_pred, binary_preds))
                tot_label = torch.cat((tot_label, total_labels[i]))

            

            # ********** CHECK RIGHT PREDICTION ON TRUE **********

            # True when it has atleast 1 right prediction on real TRUE
            # eventually --> [batch_size]
            total_1000_pred_TE = torch.empty((0), dtype=torch.bool).to(device)

            for i in range(total_1000_128_pred_TT.shape[0]):
                # single true or false tensor
                true_exist = torch.tensor([(True in total_1000_128_pred_TT[i])]).to(device) #.type(torch.bool).to(device)
                total_1000_pred_TE = torch.cat((total_1000_pred_TE, true_exist))


                writeJson(
                    prefix[i], postfix[i], label[i],
                    label_prefix[i], label_postfix[i],
                    prefix_likelihood[i], postfix_likelihood[i],
                    proj_nm
                )


            # single number count of right prediction on True from batch_size
            true_cnt_in_1000 = torch.count_nonzero(total_1000_pred_TE)
            fin_acc_TT.append((true_cnt_in_1000.item() / total_1000_128_pred_TT.shape[0])*100)


    # Compute the average accuracy and loss over the validation set.
    fin_loss = np.mean(fin_loss)
    fin_acc = np.mean(fin_acc)
    fin_acc_TT = np.mean(fin_acc_TT)

    
    print('test loss: ', fin_loss)
    print('test acc: ', fin_acc)
    print('TT acc: ', fin_acc_TT)


    # os.makedirs('../confusionMatrix/'+fn+'/')
    results = metrics.classification_report(tot_label.cpu(), tot_pred.cpu(), output_dict=True)
    results_df = pd.DataFrame.from_dict(results).transpose()
    results_df.to_excel('../confusionMatrix/'+proj_nm+'.xlsx', sheet_name='sheet1')

    print('saved precision and recall results to file!')
    
    return fin_loss, fin_acc, fin_acc_TT