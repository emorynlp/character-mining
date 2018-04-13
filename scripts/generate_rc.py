import json
from collections import defaultdict, Counter
from copy import deepcopy
import random
random.seed(1234)




def generate_padded_dataset(des_size):
    input_files = ['../../Documents/character-mining/json/friends_season_01.json',
                  '../../Documents/character-mining/json/friends_season_02.json',
                  '../../Documents/character-mining/json/friends_season_03.json',
                  '../../Documents/character-mining/json/friends_season_04.json',
                  '../../Documents/character-mining/json/friends_season_05.json',
                  '../../Documents/character-mining/json/friends_season_06.json',
                  '../../Documents/character-mining/json/friends_season_07.json',
                  '../../Documents/character-mining/json/friends_season_08.json',
                  '../../Documents/character-mining/json/friends_season_09.json',
                  '../../Documents/character-mining/json/friends_season_10.json'
                  ]

    season_samples = defaultdict(list)
    for file in input_files:
        data = json.load(open(file))
        for episode_dict in data['episodes']:
            for idx, scene_dict in enumerate(episode_dict['scenes']):
                if scene_dict['plots'] is not None:
                  
                    entities = Counter()
                    entities.update(scene_dict['rc_entities'].keys())
                
                    cur = idx
                    dialog_len = len(scene_dict['utterances'])
                    while (dialog_len < des_size and cur < len(episode_dict['scenes'])-1):
                        cur += 1
                        entities.update(episode_dict['scenes'][cur]['rc_entities'].keys())
                        dialog_len += len(episode_dict['scenes'][cur]['utterances'])
                    if dialog_len < des_size:
                        cur = idx
                        while (cur > 0 and dialog_len < des_size):
                            cur -= 1
                            entities.update(episode_dict['scenes'][cur]['rc_entities'].keys())
                            dialog_len += len(episode_dict['scenes'][cur]['utterances'])

                    masking_map = {}
                    for vi, ki in enumerate(entities.keys()):
                        masking_map[ki] = '@ent%02d' % vi

                    masked_passages = []
                    for i, passage in enumerate(scene_dict['plots']):
                        masked_sentence = []
                        ent_list = {}
                        for ent, index_list in scene_dict['rc_entities'].iteritems():
                            for index in index_list['p_ent']:
                                if i == index[0]:
                                    ent_list[index[1]] = (index[1], index[2], ent)
                        jump = 0
                        for j, token in enumerate(passage.split(' ')):
                            if jump > 0:    
                                jump -= 1
                                continue
                            if j in ent_list:
                                masked_sentence.append(masking_map[ent_list[j][2]])
                                jump = ent_list[j][1] - ent_list[j][0] - 1   
                            else:
                                masked_sentence.append(token)
                        masked_passages.append(masked_sentence)

                    cur = idx
                    dialog_len = len(scene_dict['utterances'])
                    next_dialog = []
                    while (dialog_len < des_size and cur < len(episode_dict['scenes'])-1):
                        cur += 1    
                        for i, utterance in enumerate(episode_dict['scenes'][cur]['utterances']):
                            if utterance['tokens_with_note'] is not None:
                                tokens = [w for sent in utterance['tokens_with_note'] for w in sent]
                            else:
                                tokens = [w for sent in utterance['tokens'] for w in sent]

                            masked_utter = {}
                            masked_utter['speakers'] = utterance['speakers']
                            masked_utter['tokens'] = []
                            ent_list = {}
                            for ent, index_list in episode_dict['scenes'][cur]['rc_entities'].iteritems():
                                for index in index_list['u_ent']:
                                    if i == index[0]:
                                        ent_list[index[1]] = (index[1], index[2], ent)
                                for index in index_list['s_ent']:
                                    if i == index[0]:
                                        masked_utter['speakers'][index[1]] = masking_map[ent]
                            jump = 0
                            for j, token in enumerate(tokens):
                                if jump > 0:
                                    jump -= 1
                                    continue
                                if j in ent_list:
                                    masked_utter['tokens'].append(masking_map[ent_list[j][2]])
                                    jump = ent_list[j][1] - ent_list[j][0] - 1
                                else:
                                    masked_utter['tokens'].append(token)
                            next_dialog.append(masked_utter)
                            dialog_len += 1
                            if dialog_len == des_size:
                                break

                    prev_dialog = []
                    if dialog_len < des_size:
                        cur = idx
                        while (dialog_len < des_size and cur >0):
                            cur -= 1
                            for i, utterance in enumerate(reversed(episode_dict['scenes'][cur]['utterances'])):
                                if utterance['tokens_with_note'] is not None:
                                    tokens = [w for sent in utterance['tokens_with_note'] for w in sent]
                                else:
                                    tokens = [w for sent in utterance['tokens'] for w in sent]

                                masked_utter = {}
                                masked_utter['speakers'] = utterance['speakers']
                                masked_utter['tokens'] = []
                                ent_list = {}
                                for ent, index_list in episode_dict['scenes'][cur]['rc_entities'].iteritems():
                                    for index in index_list['u_ent']:
                                        if i == len(episode_dict['scenes'][cur]['utterances'])-index[0]-1:
                                            ent_list[index[1]] = (index[1], index[2], ent)
                                    for index in index_list['s_ent']:
                                        if i == len(episode_dict['scenes'][cur]['utterances'])-index[0]-1:
                                            masked_utter['speakers'][index[1]] = masking_map[ent]
                                jump = 0
                                for j, token in enumerate(tokens):
                                    if jump > 0:
                                        jump -= 1
                                        continue
                                    if j in ent_list:
                                        masked_utter['tokens'].append(masking_map[ent_list[j][2]])
                                        jump = ent_list[j][1] - ent_list[j][0] - 1
                                    else:
                                        masked_utter['tokens'].append(token)
                                prev_dialog.append(masked_utter)
                                dialog_len += 1
                                if dialog_len == des_size:
                                    break

                    masked_dialog = []
                    for i, utterance in enumerate(scene_dict['utterances']):
                        if utterance['tokens_with_note'] is not None:
                            tokens = [w for sent in utterance['tokens_with_note'] for w in sent]          
                        else:
                            tokens = [w for sent in utterance['tokens'] for w in sent]

                        masked_utter = {}
                        masked_utter['speakers'] = utterance['speakers']
                        masked_utter['tokens'] = []
                        ent_list = {}
                        for ent, index_list in scene_dict['rc_entities'].iteritems():
                            for index in index_list['u_ent']:
                                if i == index[0]:
                                    ent_list[index[1]] = (index[1], index[2], ent)
                            for index in index_list['s_ent']:
                                if i == index[0]:
                                    masked_utter['speakers'][index[1]] = masking_map[ent]
                        jump = 0
                        for j, token in enumerate(tokens):
                            if jump > 0:
                                jump -= 1
                                continue
                            if j in ent_list:
                                masked_utter['tokens'].append(masking_map[ent_list[j][2]])
                                jump = ent_list[j][1] - ent_list[j][0] - 1  
                            else:
                                masked_utter['tokens'].append(token)
                        masked_dialog.append(masked_utter)

                    dialog_entities = Counter()
                    for ent, ent_list in scene_dict['rc_entities'].iteritems():
                        if len(ent_list['u_ent']) > 0 or len(ent_list['s_ent']) > 0:
                            dialog_entities.update([masking_map[ent]])

                    full_dialog = []
                    for u in reversed(prev_dialog):
                        full_dialog.append(u)
                    for u in masked_dialog:
                        full_dialog.append(u)
                    for u in next_dialog:
                        full_dialog.append(u)

                    for utterance in full_dialog:
                        utterance['tokens'] = ' '.join(utterance['tokens']) 
                        utterance['speakers'] = ' '.join(utterance['speakers'])

                    for sentence in masked_passages:
                        for i, token in enumerate(sentence):
                            if token.startswith('@ent') and token in dialog_entities:
                                sample = {}                                
                                query = deepcopy(sentence)
                                query[i] = '@placeholder'
                                sample['query'] = ' '.join(query)
                                sample['answer'] = token
                                sample['utterances'] = full_dialog
                                sample['scene_id'] = scene_dict['scene_id']
                                season_samples[data['season_id']].append(sample)
    
    train_samples = []
    val_samples = []
    test_samples = []
    for season_id, s_samples in season_samples.iteritems():
        l = len(s_samples)
        random.shuffle(s_samples)
        train_samples.extend(s_samples[:int(0.8*l)])
        val_samples.extend(s_samples[int(0.8 * l):int(0.9 * l)])
        test_samples.extend(s_samples[int(0.9 * l):])

    train_samples = relabel(train_samples)
    val_samples = relabel(val_samples)
    test_samples = relabel(test_samples)
    print len(train_samples)
    print len(val_samples)
    print len(test_samples)
    
    prefix = 'data_check_generated/Friends_' + str(des_size) + '_samples'
    dump_json({'train': train_samples, 'dev': val_samples, 'test': test_samples}, prefix)

def dump_json(splits, prefix):
    for split, samples in splits.iteritems():
        with open(prefix + '.' + split + '.struct.json', 'w') as fw:
            json.dump(samples, fw, indent=2)

if __name__ == '__main__':
    json_dir = '/Users/jdchoi/Git/character-mining/json'
    output_dir = '/Users/jdchoi/Git/reading-comprehension/json'
    generate_dataset(json_dir, output_dir)
    # dialog_lengths = [25, 50, 100]
    # for size in dialog_lengths:
    #     random.seed(1234)
    #     generate_padded_dataset(size)
                  