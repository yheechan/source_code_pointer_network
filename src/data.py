import json
import numpy as np
# from collections import deque
# import csv

def getTrainData(proj_list, target_project):

    total_file = 'total'

    prefix = []
    postfix = []
    label_type = []
    label_prefix = []
    label_postfix = []
    case = [] 

    for proj in proj_list:
        
        # if proj == target_project or proj == total_file: continue

        # don't remove target file for training web model
        if proj == total_file: continue

        print('Getting data for \"' + target_project + '\" from \"' + proj + '\"')

        with open('../data/' + proj, 'r') as f:
            lines = f.readlines()
        
        for line in lines:

            json_data = json.loads(line.rstrip())

            prefix.append(json_data['prefix'])
            postfix.append(json_data['postfix'])

            label_type.append(json_data['label-type'])

            label_prefix.append(json_data['label-prefix'][0])
            label_postfix.append(json_data['label-postfix'][0])

            case.append(json_data['case'])
    
        # ------------------------------------------------------
        # break for reducing test time for quick development
        # break
    
    return np.array(prefix), np.array(postfix),\
            np.array(label_type), np.array(label_prefix),\
            np.array(label_postfix), np.array(case)

def getTestData(target_project):

    prefix = []
    postfix = []
    label_type = []
    label_prefix = []
    label_postfix = []
    case = [] 

    with open('../data/' + target_project, 'r') as f:
        lines = f.readlines()
    
    for line in lines:

        json_data = json.loads(line.rstrip())

        prefix.append(json_data['prefix'])
        postfix.append(json_data['postfix'])

        label_type.append(json_data['label-type'])

        label_prefix.append(json_data['label-prefix'][0])
        label_postfix.append(json_data['label-postfix'][0])

        case.append(json_data['case'])
    
    return np.array(prefix), np.array(postfix),\
            np.array(label_type), np.array(label_prefix),\
            np.array(label_postfix), np.array(case)