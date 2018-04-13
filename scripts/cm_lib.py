# ========================================================================
# Copyright 2018 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import glob
import json
import os
import re
import random
import numpy as np
from copy import deepcopy
from collections import Counter, OrderedDict, defaultdict

__author__ = 'Jinho D. Choi'

SEASON_ID = 'season_id'
EPISODES = 'episodes'
EPISODE_ID = 'episode_id'
EPISODE = 'episode'
SCENES = 'scenes'
SCENE_ID = 'scene_id'
UTTERANCES = 'utterances'
UTTERANCE_ID = 'utterance_id'
SPEAKERS = 'speakers'
TRANSCRIPT = 'transcript'
TRANSCRIPT_WITH_NOTE = 'transcript_with_note'
TOKENS = 'tokens'
TOKENS_WITH_NOTE = 'tokens_with_note'

# character identification
CHARACTER_ENTITIES = 'character_entities'

# emotion detection
EMOTION = 'emotion'

# movie
CAPTION = 'caption'

# reading comprehension
RC_ENTITIES = 'rc_entities'
PLOTS = 'plots'
P_ENT = 'p_ent'
U_ENT = 'u_ent'
S_ENT = 's_ent'
QUERY = 'query'
ANSWER = 'answer'


# =================================== Ordered JSON ===================================

class NoIndent(object):
    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    REGEX = re.compile(r'@@@(\d+)@@@')

    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacements = {}

    def default(self, o):
        if isinstance(o, NoIndent):
            key = len(self._replacements)
            self._replacements[key] = json.dumps(o.value, **self.kwargs)
            return "@@@%d@@@" % (key)
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
        result = super(NoIndentEncoder, self).encode(o)
        out = []

        m = self.REGEX.search(result)
        while m:
            key = int(m.group(1))
            out.append(result[:m.start(0)-1])
            out.append(self._replacements[key])
            result = result[m.end(0)+1:]
            m = self.REGEX.search(result)
        return ''.join(out)


def pair(key, d, noindent=False):
    s = d[key]
    if isinstance(s, str): s = ' '.join(s.split())
    return (key, NoIndent(s)) if noindent else (key, s)


def ordered_json(input, plot=True, wo_note=True, wi_note=True, caption=True, character_entities=True, emotion=True, rc_entities=True):
    s = json.load(open(input)) if isinstance(input, str) else input
    season = OrderedDict([pair(SEASON_ID, s), pair(EPISODES, s)])
    if len(s) != len(season): print('Error: 0')
    episodes = season[EPISODES]

    for i, e in enumerate(episodes):
        episode = OrderedDict([pair(EPISODE_ID, e), pair(SCENES, e)])
        if len(e) != len(episode): print('Error: 1')
        episodes[i] = episode
        scenes = episode[SCENES]

        for j, c in enumerate(scenes):
            scene = [pair(SCENE_ID, c), pair(UTTERANCES, c)]
            if plot and PLOTS in c: scene.append(pair(PLOTS, c))
            if rc_entities and RC_ENTITIES in c:
                scene.append((RC_ENTITIES, c[RC_ENTITIES]))
                for d in c[RC_ENTITIES].values():
                    for k, v in d.items(): d[k] = NoIndent(v)
            scene = OrderedDict(scene)
            if len(c) != len(scene): print('Error 2: '+scene[SCENE_ID])
            scenes[j] = scene
            utterances = scene[UTTERANCES]

            for k, u in enumerate(utterances):
                utterance = [pair(UTTERANCE_ID, u), pair(SPEAKERS, u, True)]

                if wo_note:
                    utterance.append(pair(TRANSCRIPT, u))
                    utterance.append((TOKENS, [NoIndent(t) for t in u[TOKENS]]))
                if wi_note:
                    utterance.append(pair(TRANSCRIPT_WITH_NOTE, u))
                    twn = u[TOKENS_WITH_NOTE]
                    utterance.append((TOKENS_WITH_NOTE, [NoIndent(t) for t in twn] if twn else twn))

                if character_entities and CHARACTER_ENTITIES in u:
                    utterance.append((CHARACTER_ENTITIES, [NoIndent(t) for t in u[CHARACTER_ENTITIES]]))

                if emotion and EMOTION in u:
                    utterance.append((EMOTION, NoIndent(u[EMOTION])))

                if caption and CAPTION in u:
                    utterance.append((CAPTION, NoIndent(u[CAPTION])))

                utterance = OrderedDict(utterance)
                if len(u) != len(utterance): print('Error: 3')
                utterances[k] = utterance

    out = json.dumps(season, cls=NoIndentEncoder, indent=2)
    # out += '\n            }\n          }\n        }\n      ]\n    }\n  ]\n}'  # TODO: should not be necessary
    # out += '\n]            }\n          ]\n        }\n      ]\n    \n  ]\n}'  # character identification
    out += '\n            }\n          ]\n        }\n      ]\n    }\n  ]\n}'  # emotion detection
    return out


# =================================== General ===================================

def general_stats(json_dir):
    def stats(json_file):
        num_scenes = 0
        num_utterances = 0
        num_utterances_wn = 0
        num_sentences = 0
        num_sentences_wn = 0
        num_tokens = 0
        num_tokens_wn = 0
        speaker_list = set()

        season = json.load(open(json_file))
        episodes = season[EPISODES]

        for episode in episodes:
            scenes = episode[SCENES]
            num_scenes += len(scenes)

            for scene in scenes:
                utterances = scene[UTTERANCES]
                num_utterances_wn += len(utterances)

                for utterance in utterances:
                    speaker_list.update(utterance[SPEAKERS])

                    tokens = utterance[TOKENS]
                    if tokens:
                        num_utterances += 1
                        num_sentences += len(tokens)
                        num_tokens += sum([len(t) for t in tokens])

                    tokens_wn = utterance[TOKENS_WITH_NOTE] or tokens
                    num_sentences_wn += len(tokens_wn)
                    num_tokens_wn += sum([len(t) for t in tokens_wn])

        return [season['season_id'], len(episodes), num_scenes, num_utterances, num_sentences, num_tokens, speaker_list,
                num_utterances_wn, num_sentences_wn, num_tokens_wn]

    g_speaker_list = set()
    print('\t'.join(['Season ID', 'Episodes', 'Scenes', 'Utterances', 'Sentences', 'Tokens', 'Speakers', 'Utterances (WN)', 'Sentences (WN)', 'Tokens (WN)']))
    for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
        l = stats(json_file)
        g_speaker_list.update(l[6])
        l[6] = len(l[6])
        print('\t'.join(map(str, l)))
    print('All speakers: %s' % (len(g_speaker_list)))


def compare_peer(input_dir1, input_dir2):
    for input_file1 in sorted(glob.glob(os.path.join(input_dir1, '*.json'))):
        input_file2 = os.path.join(input_dir2, os.path.basename(input_file1))
        print(os.path.basename(input_file1))

        season1 = json.load(open(input_file1))
        season2 = json.load(open(input_file2))

        season_id = season1[SEASON_ID]
        episodes1 = season1[EPISODES]
        episodes2 = season2[EPISODES]
        if len(episodes1) != len(episodes2):
            print('Episode mismatch: %s - %d, %d' % (season_id, len(episodes1), len(episodes2)))

        for episode1, episode2 in zip(episodes1, episodes2):
            episode_id = episode1[EPISODE_ID]
            scenes1 = episode1[SCENES]
            scenes2 = episode2[SCENES]
            if len(scenes1) != len(scenes2):
                print('Scene mismatch: %s - %d, %d' % (episode_id, len(scenes1), len(scenes2)))

            for scene1, scene2 in zip(scenes1, scenes2):
                scene_id = scene1[SCENE_ID]
                utterances1 = scene1[UTTERANCES]
                utterances2 = scene2[UTTERANCES]
                if len(utterances1) != len(utterances2):
                    print('Utterance mismatch: %s - %d, %d' % (scene_id, len(utterances1), len(utterances2)))

                for utterance1, utterance2 in zip(utterances1, utterances2):
                    utterance_id = utterance1[UTTERANCE_ID]
                    tokens1 = utterance1[TOKENS]
                    tokens2 = utterance2[TOKENS]
                    if len(tokens1) != len(tokens2):
                        print('Token mismatch: %s - %d, %d' % (utterance_id, len(tokens1), len(tokens2)))

                    m = [i for i in range(len(tokens1)) if tokens1[i] != tokens2[i]]
                    if m:
                        print('Token mismatch: %s - %s' % (utterance_id, str(m)))

                    tokens1 = utterance1[TOKENS_WITH_NOTE]
                    tokens2 = utterance2[TOKENS_WITH_NOTE]

                    if tokens1 is None and tokens2 is None:
                        continue

                    if len(tokens1) != len(tokens2):
                        print('Token WN mismatch: %s - %d, %d' % (utterance_id, len(tokens1), len(tokens2)))

                    m = [i for i in range(len(tokens1)) if tokens1[i] != tokens2[i]]
                    if m:
                        print('Token WN mismatch: %s - %s' % (utterance_id, str(m)))


# =================================== Character Identification ===================================

def extract_character_identification(input_dir, output_dir):
    """
    trn: episodes 1-19
    dev: episodes 20-21
    tst: episodes 22-end
    """
    trn = {SEASON_ID: 'trn', EPISODES: []}
    dev = {SEASON_ID: 'dev', EPISODES: []}
    tst = {SEASON_ID: 'tst', EPISODES: []}

    def get_entities(entity_list):
        return [entity for entity in entity_list if entity[-1] != 'Non-Entity']

    for i, input_file in enumerate(sorted(glob.glob(os.path.join(input_dir, '*.json')))):
        if i >= 4: break
        season = json.load(open(input_file))
        print(input_file)

        for episode in season[EPISODES]:
            episode_id = int(episode[EPISODE_ID].split('_')[1][1:])
            d = tst if episode_id >= 22 else dev if episode_id >= 20 else trn
            d[EPISODES].append(episode)
            scenes = []

            for scene in episode[SCENES]:
                utterances = []

                for utterance in scene[UTTERANCES]:
                    if utterance[TOKENS]:
                        utterances.append(utterance)

                        if CHARACTER_ENTITIES in utterance:
                            utterance[CHARACTER_ENTITIES] = [get_entities(entity_list) for entity_list in utterance[CHARACTER_ENTITIES]]
                        else:
                            print(utterance[UTTERANCE_ID])

                if utterances:
                    scene[UTTERANCES] = utterances
                    scenes.append(scene)

            episode[SCENES] = scenes

    with open(os.path.join(output_dir, 'character-identification-trn.json'), 'w') as fout:
        fout.write(ordered_json(trn, plot=False, wi_note=False, caption=False, emotion=False, rc_entities=False))

    with open(os.path.join(output_dir, 'character-identification-dev.json'), 'w') as fout:
        fout.write(ordered_json(dev, plot=False, wi_note=False, caption=False, emotion=False, rc_entities=False))

    with open(os.path.join(output_dir, 'character-identification-tst.json'), 'w') as fout:
        fout.write(ordered_json(tst, plot=False, wi_note=False, caption=False, emotion=False, rc_entities=False))


def entity_stats(json_dir):
    def stats(json_file):
        speaker_list = []
        entity_list = []
        num_scenes = 0
        num_utterances = 0
        num_tokens = 0
        num_mentions = 0

        season = json.load(open(json_file))
        episodes = season[EPISODES]

        for episode in episodes:
            scenes = episode[SCENES]
            num_scenes += len(scenes)

            for scene in scenes:
                utterances = scene[UTTERANCES]
                num_utterances += len(utterances)

                for utterance in utterances:
                    num_tokens += sum([len(t) for t in utterance[TOKENS]])
                    speaker_list.extend(utterance[SPEAKERS])

                    if len(utterance[TOKENS]) != len(utterance[CHARACTER_ENTITIES]):
                        print(utterances[UTTERANCE_ID])

                    for character_entities in utterance[CHARACTER_ENTITIES]:
                        num_mentions += len(character_entities)
                        for entities in character_entities:
                            entity_list.extend(entities[2:])

        g_speaker_list.extend(speaker_list)
        g_entity_list.extend(entity_list)
        return [season[SEASON_ID], len(episodes), num_scenes, num_utterances, num_tokens, len(set(speaker_list)), num_mentions, len(set(entity_list))]

    g_speaker_list = []
    g_entity_list = []
    print('\t'.join(['Dataset', 'Episodes', 'Scenes', 'Utterances', 'Tokens', 'Speakers', 'Mentions', 'Entities']))

    for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
        l = stats(json_file)
        print('\t'.join(map(str, l)))

    print('All speakers: %s' % (len(set(g_speaker_list))))
    print('All entities: %s' % (len(set(g_entity_list))))


# =================================== Emotion Detection ===================================

def extract_emotion_detection(input_dir, output_dir):
    trn = {SEASON_ID: 'trn', EPISODES: []}
    dev = {SEASON_ID: 'dev', EPISODES: []}
    tst = {SEASON_ID: 'tst', EPISODES: []}

    DEV = {'s01_e15', 's01_e20', 's02_e10', 's02_e20', 's03_e01', 's03_e09', 's03_e21', 's04_e01', 's04_e06', 's04_e10', 's04_e21'}
    TST = {'s01_e01', 's01_e10', 's02_e08', 's02_e23', 's03_e08', 's03_e20', 's04_e02', 's04_e20', 's04_e17'}

    def get_entities(entity_list):
        return [entity for entity in entity_list if entity[-1] != 'Non-Entity']

    for i, input_file in enumerate(sorted(glob.glob(os.path.join(input_dir, '*.json')))):
        if i >= 4: break
        season = json.load(open(input_file))
        print(input_file)

        for episode in season[EPISODES]:
            episode_id = episode[EPISODE_ID]
            d = tst if episode_id in TST else dev if episode_id in DEV else trn
            d[EPISODES].append(episode)
            scenes = []

            for scene in episode[SCENES]:
                utterances = []
                emotions = 0
                misses = []

                for utterance in scene[UTTERANCES]:
                    if utterance[TOKENS]:
                        if EMOTION in utterance:
                            utterance[EMOTION] = utterance[EMOTION][0]
                            emotions += 1
                        else:
                            misses.append(utterance[UTTERANCE_ID])

                        utterances.append(utterance)

                if emotions > 0:
                    if emotions != len(utterances): print(misses)
                    scene[UTTERANCES] = utterances
                    scenes.append(scene)

            episode[SCENES] = scenes

    with open(os.path.join(output_dir, 'emotion-detection-trn.json'), 'w') as fout:
        fout.write(ordered_json(trn, plot=False, wi_note=False, caption=False, character_entities=False, rc_entities=False))

    with open(os.path.join(output_dir, 'emotion-detection-dev.json'), 'w') as fout:
        fout.write(ordered_json(dev, plot=False, wi_note=False, caption=False, character_entities=False, rc_entities=False))

    with open(os.path.join(output_dir, 'emotion-detection-tst.json'), 'w') as fout:
        fout.write(ordered_json(tst, plot=False, wi_note=False, caption=False, character_entities=False, rc_entities=False))


def emotion_stats(json_dir):
    def stats(json_file):
        emotions = {}
        num_scenes = 0
        num_utterances = 0
        episode_ids = []

        season = json.load(open(json_file))
        episodes = season[EPISODES]

        for episode in episodes:
            episode_ids.append(episode[EPISODE_ID])
            scenes = episode[SCENES]
            num_scenes += len(scenes)

            for scene in scenes:
                utterances = scene[UTTERANCES]
                num_utterances += len(utterances)

                for utterance in utterances:
                    e = utterance[EMOTION]
                    emotions[e] = emotions.setdefault(e, 0) + 1

        print(episode_ids)
        return [season[SEASON_ID], len(episodes), num_scenes, num_utterances] + [emotions[e] for e in emotion_list]

    emotion_list = ['Joyful', 'Mad', 'Neutral', 'Peaceful', 'Powerful', 'Sad', 'Scared']
    print('\t'.join(['Dataset', 'Episodes', 'Scenes', 'Utterances'] + emotion_list))

    for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
        l = stats(json_file)
        print('\t'.join(map(str, l)))


# =================================== Reading Comprehension ===================================

def relabel(samples):
    re_samples = []
    for sample in samples:
        sam = {}

        q_words = sample[QUERY].split(' ')
        d_words = []
        for utter in sample[UTTERANCES]:
            d_words += utter[SPEAKERS]
            d_words += utter[TOKENS]

        entity_dict = {}
        entity_id = 0
        for word in d_words + q_words:
            if (word.startswith('@ent')) and (word not in entity_dict):
                entity_dict[word] = '@ent%02d' % entity_id
                entity_id += 1

        re_document = []
        for utter in sample[UTTERANCES]:
            sent = {SPEAKERS: ' '.join(
                [entity_dict[w] if w in entity_dict else w for w in utter[SPEAKERS]]),
                TOKENS: ' '.join([entity_dict[w] if w in entity_dict else w for w in utter[TOKENS]])}
            re_document.append(sent)

        sam[SCENE_ID] = sample[SCENE_ID]
        sam[QUERY] = ' '.join([entity_dict[w] if w in entity_dict else w for w in q_words])
        sam[ANSWER] = entity_dict[sample[ANSWER]]
        sam[UTTERANCES] = re_document
        re_samples.append(sam)
    return re_samples


def extract_reading_comprehension(json_dir, output_dir):
    season_samples = defaultdict(list)
    random.seed(1234)

    for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
        season = json.load(open(json_file))
        for episode in season[EPISODES]:
            for scene in episode[SCENES]:
                if PLOTS in scene and scene[PLOTS]:
                    masking_map = {}
                    for vi, ki in enumerate(scene[RC_ENTITIES].keys()):
                        masking_map[ki] = '@ent%02d' % vi

                    masked_passages = []
                    for i, passage in enumerate(scene[PLOTS]):
                        masked_sentence = []
                        ent_list = {}
                        for ent, index_list in scene[RC_ENTITIES].items():
                            for index in index_list[P_ENT]:
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

                    masked_dialog = []
                    for i, utterance in enumerate(scene[UTTERANCES]):
                        if utterance[TOKENS_WITH_NOTE] is not None:
                            tokens = [w for sent in utterance[TOKENS_WITH_NOTE] for w in sent]
                        else:
                            tokens = [w for sent in utterance[TOKENS] for w in sent]

                        masked_utter = {SPEAKERS: utterance[SPEAKERS], TOKENS: []}
                        ent_list = {}
                        for ent, index_list in scene[RC_ENTITIES].items():
                            for index in index_list[U_ENT]:
                                if i == index[0]:
                                    ent_list[index[1]] = (index[1], index[2], ent)
                            for index in index_list[S_ENT]:
                                if i == index[0]:
                                    masked_utter[SPEAKERS][index[1]] = masking_map[ent]

                        jump = 0
                        for j, token in enumerate(tokens):
                            if jump > 0:
                                jump -= 1
                                continue
                            if j in ent_list:
                                masked_utter[TOKENS].append(masking_map[ent_list[j][2]])
                                jump = ent_list[j][1] - ent_list[j][0] - 1
                            else:
                                masked_utter[TOKENS].append(token)
                        masked_dialog.append(masked_utter)

                    dialog_entities = Counter()
                    for ent, ent_list in scene[RC_ENTITIES].items():
                        if len(ent_list[U_ENT]) > 0 or len(ent_list[S_ENT]) > 0:
                            dialog_entities.update([masking_map[ent]])

                    for sentence in masked_passages:
                        for i, token in enumerate(sentence):
                            if token.startswith('@ent') and token in dialog_entities:
                                sample = {}
                                query = deepcopy(sentence)
                                query[i] = '@placeholder'
                                sample[QUERY] = ' '.join(query)
                                sample[ANSWER] = token
                                sample[UTTERANCES] = masked_dialog
                                sample[SCENE_ID] = scene[SCENE_ID]
                                season_samples[season[SEASON_ID]].append(sample)

    trn = []
    dev = []
    tst = []
    for season_id, s_samples in season_samples.items():
        n = len(s_samples)
        random.shuffle(s_samples)
        trn.extend(s_samples[:int(0.8 * n)])
        dev.extend(s_samples[int(0.8 * n):int(0.9 * n)])
        tst.extend(s_samples[int(0.9 * n):])

    trn = relabel(trn)
    dev = relabel(dev)
    tst = relabel(tst)

    with open(os.path.join(output_dir, 'trn.json'), 'w') as fout:
        fout.write(json.dumps(trn, indent=2))

    with open(os.path.join(output_dir, 'dev.json'), 'w') as fout:
        fout.write(json.dumps(dev, indent=2))

    with open(os.path.join(output_dir, 'tst.json'), 'w') as fout:
        fout.write(json.dumps(tst, indent=2))


def reading_stats(json_dir):
    def create(dataset, num_queries, num_entity_count_query, num_entity_type_query, num_entity_count_utt, num_entity_type_utt, num_utterances):
        return [dataset,
                num_queries,
                num_utterances / num_queries,
                num_entity_type_query / num_queries,
                num_entity_count_query / num_queries,
                num_entity_type_utt / num_queries,
                num_entity_count_utt / num_queries]

    def stats(json_file):
        documents = json.load(open(json_file))
        num_queries = len(documents)
        num_entity_count_query = 0
        num_entity_type_query = 0
        num_entity_count_utt = 0
        num_entity_type_utt = 0
        num_utterances = 0

        for doc in documents:
            ents = [doc[ANSWER] if q == '@placeholder' else q for q in doc[QUERY].split() if q.startswith('@ent') or q == '@placeholder']
            num_entity_count_query += len(ents)
            num_entity_type_query += len(set(ents))

            num_utterances += len(doc[UTTERANCES])
            ents = []

            for utterance in doc[UTTERANCES]:
                ents.extend(utterance[SPEAKERS].split())
                ents.extend([t for t in utterance[TOKENS].split() if t.startswith('@ent')])

            num_entity_type_utt += len(set(ents))
            num_entity_count_utt += len(ents)

        return [num_queries, num_entity_count_query, num_entity_type_query, num_entity_count_utt, num_entity_type_utt, num_utterances]

    print('\t'.join(['Dataset', 'Queries', 'U / Q', '{E} / Q', '[E] / Q', '{E} / U', '[E] / U']))
    g_num = np.zeros(6)

    for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
        l = stats(json_file)
        g_num += np.array(l)
        print('\t'.join(map(str, create(json_file[-15:-12].upper(), *l))))

    print('\t'.join(map(str, create('Total', *g_num))))



# =================================== Main ===================================

if __name__ == '__main__':
    # json_dir = '/Users/jdchoi/Git/character-mining/json'
    # general_stats(json_dir)

    # character identification
    # input_dir = '/Users/jdchoi/Git/character-mining/json'
    # output_dir = '/Users/jdchoi/Git/character-identification/json'
    # extract_character_identification(input_dir, output_dir)
    # entity_stats(output_dir)

    # emotino detection
    # input_dir = '/Users/jdchoi/Git/character-mining/json'
    # output_dir = '/Users/jdchoi/Git/emotion-detection/json'
    # extract_emotion_detection(input_dir, output_dir)
    # emotion_stats(output_dir)

    # reading comprehension
    json_dir = '/Users/jdchoi/Git/character-mining/json'
    output_dir = '/Users/jdchoi/Git/reading-comprehension/json'
    # extract_reading_comprehension(json_dir, output_dir)
    reading_stats(output_dir)



    # input_dir = '/Users/jdchoi/Git/character-mining/json'
    # ann_dir = '/Users/jdchoi/Downloads/dataset'
    # output_dir = '/Users/jdchoi/Git/character-mining/json/em'
    # merge_em(input_dir, ann_dir, output_dir)

    # input_dir1 = '/Users/jdchoi/Git/character-mining-dev/json-bak'
    # input_dir2 = '/Users/jdchoi/Downloads/Friends_newly_compiled'
    # output_dir = '/Users/jdchoi/Git/character-mining/json'
    # merge_rc(input_dir1, input_dir2, output_dir)










# def merge_rc(input_dir1, input_dir2, output_dir):
#     def get_entities(rc_entities):
#         plot = rc_entities['plot_entities']
#         speaker = rc_entities['speaker_entities']
#         utterance = rc_entities['utterance_entities']
#         entities = {}
#
#         if plot:
#             for name, ts in plot.items():
#                 d = entities.setdefault(name, OrderedDict([(P_ENT, []), (U_ENT, []), (S_ENT, [])]))
#                 d[P_ENT] = [t[:-1] for t in ts]
#
#         for name, ts in utterance.items():
#             d = entities.setdefault(name, OrderedDict([(P_ENT, []), (U_ENT, []), (S_ENT, [])]))
#             d[U_ENT] = [t[:-1] for t in ts]
#
#         for name, ts in speaker.items():
#             d = entities.setdefault(name, OrderedDict([(P_ENT, []), (U_ENT, []), (S_ENT, [])]))
#             d[S_ENT] = [t[:-1] for t in ts]
#
#         return entities
#
#     for input_file1 in sorted(glob.glob(os.path.join(input_dir1, '*.json'))):
#         input_file2 = os.path.join(input_dir2, os.path.basename(input_file1))
#         print(os.path.basename(input_file1))
#
#         season1 = json.load(open(input_file1))
#         season2 = json.load(open(input_file2))
#
#         episodes1 = season1[EPISODES]
#         episodes2 = season2[EPISODES]
#
#         for episode1, episode2 in zip(episodes1, episodes2):
#             scenes1 = episode1[SCENES]
#             scenes2 = episode2[SCENES]
#
#             for scene1, scene2 in zip(scenes1, scenes2):
#                 scene1[PLOTS] = scene2[PLOTS]
#                 scene1[RC_ENTITIES] = get_entities(scene2[RC_ENTITIES])
#
#         with open(os.path.join(output_dir, os.path.basename(input_file1)), 'w') as fout:
#             fout.write(ordered_json(season1))
#
#
# def merge_em(input_dir, ann_dir, output_dir):
#     def extend_ann(ann_file, ls):
#         fin = open(ann_file)
#
#         for i, line in enumerate(fin):
#             if i == 0: continue
#             l = line.split()
#             season_id = int(l[0]) - 1
#             episode_id = int(l[1]) - 1
#             scene_id = int(l[2]) - 1
#             utterance_id = int(l[3])
#             annotation = l[4:8]
#             gold = l[10]
#             ls.append((season_id, episode_id, scene_id, utterance_id, annotation, gold))
#
#
#     annotations = []
#     for ann_file in glob.glob(os.path.join(ann_dir, '*.tsv')): extend_ann(ann_file, annotations)
#     seasons = [json.load(open(input_file)) for input_file in sorted(glob.glob(os.path.join(input_dir, '*.json')))]
#
#     for season_id, episode_id, scene_id, utterance_id, annotation, gold in annotations:
#         utterance = seasons[season_id][EPISODES][episode_id][SCENES][scene_id][UTTERANCES][utterance_id]
#         if EMOTION in utterance:
#             if utterance[EMOTION] != gold: print(utterance[UTTERANCE_ID])
#             utterance[EMOTION] = [gold, annotation]
#         else:
#             print(utterance[UTTERANCE_ID])
#
#     for i, season in enumerate(seasons):
#         with open(os.path.join(output_dir, 'friends_season_0%d.json' % (i+1)), 'w') as fout:
#             fout.write(ordered_json(season))
#
# def extract_reading_comprehension_padded(json_dir, output_dir, des_size):
#     season_samples = defaultdict(list)
#     random.seed(1234)
#
#     for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
#         data = json.load(open(json_file))
#         for episode_dict in data[EPISODES]:
#             for idx, scene_dict in enumerate(episode_dict[SCENES]):
#                 if scene_dict[PLOTS] is not None:
#
#                     entities = Counter()
#                     entities.update(scene_dict[RC_ENTITIES].keys())
#
#                     cur = idx
#                     dialog_len = len(scene_dict[UTTERANCES])
#                     while dialog_len < des_size and cur < len(episode_dict[SCENES]) - 1:
#                         cur += 1
#                         entities.update(episode_dict[SCENES][cur][RC_ENTITIES].keys())
#                         dialog_len += len(episode_dict[SCENES][cur][UTTERANCES])
#                     if dialog_len < des_size:
#                         cur = idx
#                         while (cur > 0 and dialog_len < des_size):
#                             cur -= 1
#                             entities.update(episode_dict[SCENES][cur][RC_ENTITIES].keys())
#                             dialog_len += len(episode_dict[SCENES][cur][UTTERANCES])
#
#                     masking_map = {}
#                     for vi, ki in enumerate(entities.keys()):
#                         masking_map[ki] = '@ent%02d' % vi
#
#                     masked_passages = []
#                     for i, passage in enumerate(scene_dict[PLOTS]):
#                         masked_sentence = []
#                         ent_list = {}
#                         for ent, index_list in scene_dict[RC_ENTITIES].items():
#                             for index in index_list[P_ENT]:
#                                 if i == index[0]:
#                                     ent_list[index[1]] = (index[1], index[2], ent)
#                         jump = 0
#                         for j, token in enumerate(passage.split(' ')):
#                             if jump > 0:
#                                 jump -= 1
#                                 continue
#                             if j in ent_list:
#                                 masked_sentence.append(masking_map[ent_list[j][2]])
#                                 jump = ent_list[j][1] - ent_list[j][0] - 1
#                             else:
#                                 masked_sentence.append(token)
#                         masked_passages.append(masked_sentence)
#
#                     cur = idx
#                     dialog_len = len(scene_dict[UTTERANCES])
#                     next_dialog = []
#                     while dialog_len < des_size and cur < len(episode_dict[SCENES]) - 1:
#                         cur += 1
#                         for i, utterance in enumerate(episode_dict[SCENES][cur][UTTERANCES]):
#                             if utterance[TOKENS_WITH_NOTE] is not None:
#                                 tokens = [w for sent in utterance[TOKENS_WITH_NOTE] for w in sent]
#                             else:
#                                 tokens = [w for sent in utterance[TOKENS] for w in sent]
#
#                             masked_utter = {SPEAKERS: utterance[SPEAKERS], TOKENS: []}
#                             ent_list = {}
#                             for ent, index_list in episode_dict[SCENES][cur][RC_ENTITIES].items():
#                                 for index in index_list[U_ENT]:
#                                     if i == index[0]:
#                                         ent_list[index[1]] = (index[1], index[2], ent)
#                                 for index in index_list[S_ENT]:
#                                     if i == index[0]:
#                                         masked_utter[SPEAKERS][index[1]] = masking_map[ent]
#                             jump = 0
#                             for j, token in enumerate(tokens):
#                                 if jump > 0:
#                                     jump -= 1
#                                     continue
#                                 if j in ent_list:
#                                     masked_utter[TOKENS].append(masking_map[ent_list[j][2]])
#                                     jump = ent_list[j][1] - ent_list[j][0] - 1
#                                 else:
#                                     masked_utter[TOKENS].append(token)
#                             next_dialog.append(masked_utter)
#                             dialog_len += 1
#                             if dialog_len == des_size:
#                                 break
#
#                     prev_dialog = []
#                     if dialog_len < des_size:
#                         cur = idx
#                         while dialog_len < des_size and cur > 0:
#                             cur -= 1
#                             for i, utterance in enumerate(reversed(episode_dict[SCENES][cur][UTTERANCES])):
#                                 if utterance[TOKENS_WITH_NOTE] is not None:
#                                     tokens = [w for sent in utterance[TOKENS_WITH_NOTE] for w in sent]
#                                 else:
#                                     tokens = [w for sent in utterance[TOKENS] for w in sent]
#
#                                 masked_utter = {}
#                                 masked_utter[SPEAKERS] = utterance[SPEAKERS]
#                                 masked_utter[TOKENS] = []
#                                 ent_list = {}
#                                 for ent, index_list in episode_dict[SCENES][cur][RC_ENTITIES].items():
#                                     for index in index_list[U_ENT]:
#                                         if i == len(episode_dict[SCENES][cur][UTTERANCES]) - index[0] - 1:
#                                             ent_list[index[1]] = (index[1], index[2], ent)
#                                     for index in index_list[S_ENT]:
#                                         if i == len(episode_dict[SCENES][cur][UTTERANCES]) - index[0] - 1:
#                                             masked_utter[SPEAKERS][index[1]] = masking_map[ent]
#                                 jump = 0
#                                 for j, token in enumerate(tokens):
#                                     if jump > 0:
#                                         jump -= 1
#                                         continue
#                                     if j in ent_list:
#                                         masked_utter[TOKENS].append(masking_map[ent_list[j][2]])
#                                         jump = ent_list[j][1] - ent_list[j][0] - 1
#                                     else:
#                                         masked_utter[TOKENS].append(token)
#                                 prev_dialog.append(masked_utter)
#                                 dialog_len += 1
#                                 if dialog_len == des_size:
#                                     break
#
#                     masked_dialog = []
#                     for i, utterance in enumerate(scene_dict[UTTERANCES]):
#                         if utterance[TOKENS_WITH_NOTE] is not None:
#                             tokens = [w for sent in utterance[TOKENS_WITH_NOTE] for w in sent]
#                         else:
#                             tokens = [w for sent in utterance[TOKENS] for w in sent]
#
#                         masked_utter = {SPEAKERS: utterance[SPEAKERS], TOKENS: []}
#                         ent_list = {}
#                         for ent, index_list in scene_dict[RC_ENTITIES].items():
#                             for index in index_list[U_ENT]:
#                                 if i == index[0]:
#                                     ent_list[index[1]] = (index[1], index[2], ent)
#                             for index in index_list[S_ENT]:
#                                 if i == index[0]:
#                                     masked_utter[SPEAKERS][index[1]] = masking_map[ent]
#                         jump = 0
#                         for j, token in enumerate(tokens):
#                             if jump > 0:
#                                 jump -= 1
#                                 continue
#                             if j in ent_list:
#                                 masked_utter[TOKENS].append(masking_map[ent_list[j][2]])
#                                 jump = ent_list[j][1] - ent_list[j][0] - 1
#                             else:
#                                 masked_utter[TOKENS].append(token)
#                         masked_dialog.append(masked_utter)
#
#                     dialog_entities = Counter()
#                     for ent, ent_list in scene_dict[RC_ENTITIES].items():
#                         if len(ent_list[U_ENT]) > 0 or len(ent_list[S_ENT]) > 0:
#                             dialog_entities.update([masking_map[ent]])
#
#                     full_dialog = []
#                     for u in reversed(prev_dialog):
#                         full_dialog.append(u)
#                     for u in masked_dialog:
#                         full_dialog.append(u)
#                     for u in next_dialog:
#                         full_dialog.append(u)
#
#                     for sentence in masked_passages:
#                         for i, token in enumerate(sentence):
#                             if token.startswith('@ent') and token in dialog_entities:
#                                 sample = {}
#                                 query = deepcopy(sentence)
#                                 query[i] = '@placeholder'
#                                 sample[QUERY] = ' '.join(query)
#                                 sample[ANSWER] = token
#                                 sample[UTTERANCES] = full_dialog
#                                 sample[SCENE_ID] = scene_dict[SCENE_ID]
#                                 season_samples[data[SEASON_ID]].append(sample)
#
#     trn = []
#     dev = []
#     tst = []
#     for season_id, s_samples in season_samples.items():
#         l = len(s_samples)
#         random.shuffle(s_samples)
#         trn.extend(s_samples[:int(0.8 * l)])
#         dev.extend(s_samples[int(0.8 * l):int(0.9 * l)])
#         tst.extend(s_samples[int(0.9 * l):])
#
#     trn = relabel(trn)
#     dev = relabel(dev)
#     tst = relabel(tst)
#     print(len(trn), len(dev), len(tst))
#
#     with open(os.path.join(output_dir, 'trn-%d.json' % des_size), 'w') as fout:
#         fout.write(json.dumps(trn, indent=2))
#
#     with open(os.path.join(output_dir, 'dev-%d.json' % des_size), 'w') as fout:
#         fout.write(json.dumps(dev, indent=2))
#
#     with open(os.path.join(output_dir, 'tst-%d.json' % des_size), 'w') as fout:
#         fout.write(json.dumps(tst, indent=2))