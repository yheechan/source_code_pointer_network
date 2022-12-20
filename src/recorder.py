import json

def recordData(proj_list, version):

    token_choices = []
    max_len = -1

    prefix = []
    postfix = []
    # label_type = []
    # label_prefix = []
    # label_postfix = []
    # case = []

    total_ones_prefix = {}
    total_ones_postfix = {}
    project_data = []

    for fn in proj_list:
        print('checking project \"' + fn + '\"')

        with open('../data/'+version+'/'+fn, 'r') as f:
            lines = f.readlines()
        
        for line in lines:

            json_data = json.loads(line)

            if int(json_data['label-type'][0]) not in token_choices:
                print('added choice \"' + str(json_data['label-type'][0]) + '\"')
                token_choices.append(int(json_data['label-type'][0]))

            for tok in json_data['prefix']:
                if int(tok) not in token_choices:
                    print('added choice \"' + str(tok) + '\"')
                    token_choices.append(int(tok))
            
            for tok in json_data['postfix']:
                if int(tok) not in token_choices:
                    print('added choice \"' + str(tok) + '\"')
                    token_choices.append(int(tok))
            
            if len(json_data['prefix']) > max_len: max_len = len(json_data['prefix'])
            if len(json_data['postfix']) > max_len: max_len = len(json_data['postfix'])

            for pointer_labels in json_data['label-prefix']:
                cnt = pointer_labels.count(1)

                if cnt not in total_ones_prefix:
                    total_ones_prefix[cnt] = 1
                else:
                    total_ones_prefix[cnt] += 1

            for pointer_labels in json_data['label-postfix']:
                cnt = pointer_labels.count(1)

                if cnt not in total_ones_postfix:
                    total_ones_postfix[cnt] = 1
                else:
                    total_ones_postfix[cnt] += 1

        project_data.append(fn + ': \t\t' + str(len(lines)))

    # token_choices.append('213')

    token_choices.sort()
    token_choices = list(map(str, token_choices))
    
    record_list('../record/project_data', project_data)
    record_list('../record/token_choices', token_choices)
    record_list('../record/max_len', [str(max_len)])
    record_dict('../record/total_ones_prefix', total_ones_prefix)
    record_dict('../record/total_ones_postfix', total_ones_postfix)

def record_list(fn, list):
    print('writing to \"' + fn + '\"')

    with open(fn, 'w') as f:
        f.write('\n'.join(list))

def record_dict(fn, dict):
    print('writing to \"' + fn + '\"')

    with open(fn, 'w') as f:
        for key in dict.keys():
            s = str(key) + ': ' + str(dict[key]) + '\n'
            f.write(s)