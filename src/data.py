import json
import numpy as np
# from collections import deque
# import csv

def get8ByteList(ids):
    total_byte_list = []

    for word in ids:
        single_byte_list = []

        for char in word:
            if len(single_byte_list) == 8: break
            single_byte_list.append(ord(char))
        
        single_byte_list += [0] * (8-len(single_byte_list))
        total_byte_list.append(np.array(single_byte_list))

    return np.array(total_byte_list)

def getTrainData(proj_list, target_project, version):

    total_file = 'total'

    prefix = []
    prefix_ids = []
    postfix = []
    postfix_ids = []
    label_type = []
    label_prefix = []
    label_postfix = []
    case = [] 

    for proj in proj_list:
        
        # if proj == target_project or proj == total_file: continue
        # don't remove target file for training web model
        if proj == total_file: continue

        print('Getting data for \"' + target_project + '\" from \"' + proj + '\"')

        with open('../data/' + version + '/' + proj, 'r') as f:
            lines = f.readlines()
        
        for line in lines:

            json_data = json.loads(line.rstrip())

            prefix.append(json_data['prefix'])
            prefix_ids.append(get8ByteList(json_data['prefix-ids']))

            postfix.append(json_data['postfix'])
            postfix_ids.append(get8ByteList(json_data['postfix-ids']))

            label_type.append(json_data['label-type'])

            label_prefix.append(json_data['label-prefix'][0])
            label_postfix.append(json_data['label-postfix'][0])

            case.append(json_data['case'])
    
        # ------------------------------------------------------
        # break for reducing test time for quick development
        break
    
    return np.array(prefix), np.array(prefix_ids),\
            np.array(postfix), np.array(postfix_ids),\
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