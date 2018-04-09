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
import numpy as np
from collections import Counter, OrderedDict

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
PLOTS = 'plots'
RC_ENTITIES = 'rc_entities'
P_ENT = 'p_ent'
U_ENT = 'u_ent'
S_ENT = 's_ent'

# ============================== UTILITIES ==============================


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
            # if len(c) != len(scene): print('Error 2: '+scene[SCENE_ID])
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
                    utterance.append(pair(EMOTION, u))

                if caption and CAPTION in u:
                    utterance.append((CAPTION, NoIndent(u[CAPTION])))

                utterance = OrderedDict(utterance)
                # if len(u) != len(utterance): print('Error: 3')
                utterances[k] = utterance

    out = json.dumps(season, cls=NoIndentEncoder, indent=2)
    out += '\n            }\n          }\n        }\n      ]\n    }\n  ]\n}'  # TODO: should not be necessary
    # out += '\n]            }\n          ]\n        }\n      ]\n    \n  ]\n}'  # character identification
    return out


# ============================== STATISTICS ==============================

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

                    for character_entities in utterance[CHARACTER_ENTITIES]:
                        num_mentions += len(character_entities)
                        for entities in character_entities:
                            entity_list.extend(entities[2:])

        g_speaker_list.extend(speaker_list)
        g_entity_list.extend(entity_list)
        return [season[SEASON_ID], len(episodes), num_scenes, num_utterances, num_tokens, len(speaker_list), num_mentions, len(entity_list)]

    g_speaker_list = []
    g_entity_list = []
    print('\t'.join(['Dataset', 'Episodes', 'Scenes', 'Utterances', 'Tokens', 'Speakers', 'Mentions', 'Entities']))

    for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
        l = stats(json_file)
        print('\t'.join(map(str, l)))

    print('All speakers: %s' % (len(set(g_speaker_list))))
    print('All entities: %s' % (len(set(g_entity_list))))


# ============================== EXTRACTION ==============================

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


def merge_rc(input_dir1, input_dir2, output_dir):
    def get_entities(rc_entities):
        plot = rc_entities['plot_entities']
        speaker = rc_entities['speaker_entities']
        utterance = rc_entities['utterance_entities']
        entities = {}

        if plot:
            for name, ts in plot.items():
                d = entities.setdefault(name, OrderedDict([(P_ENT, []), (U_ENT, []), (S_ENT, [])]))
                d[P_ENT] = [t[:-1] for t in ts]

        for name, ts in utterance.items():
            d = entities.setdefault(name, OrderedDict([(P_ENT, []), (U_ENT, []), (S_ENT, [])]))
            d[U_ENT] = [t[:-1] for t in ts]

        for name, ts in speaker.items():
            d = entities.setdefault(name, OrderedDict([(P_ENT, []), (U_ENT, []), (S_ENT, [])]))
            d[S_ENT] = [t[:-1] for t in ts]

        return entities

    for input_file1 in sorted(glob.glob(os.path.join(input_dir1, '*.json'))):
        input_file2 = os.path.join(input_dir2, os.path.basename(input_file1))
        print(os.path.basename(input_file1))

        season1 = json.load(open(input_file1))
        season2 = json.load(open(input_file2))

        episodes1 = season1[EPISODES]
        episodes2 = season2[EPISODES]

        for episode1, episode2 in zip(episodes1, episodes2):
            scenes1 = episode1[SCENES]
            scenes2 = episode2[SCENES]

            for scene1, scene2 in zip(scenes1, scenes2):
                scene1[PLOTS] = scene2[PLOTS]
                scene1[RC_ENTITIES] = get_entities(scene2[RC_ENTITIES])

        with open(os.path.join(output_dir, os.path.basename(input_file1)), 'w') as fout:
            fout.write(ordered_json(season1))






if __name__ == '__main__':
    json_dir = '/Users/jdchoi/Git/character-mining/json'
    general_stats(json_dir)

    # input_dir = '/Users/jdchoi/Git/character-mining/json'
    # output_dir = '/Users/jdchoi/Git/character-identification/json'
    # extract_character_identification(input_dir, output_dir)
    # entity_stats(output_dir)

    # entity_stats(json_dir)

    # input_dir1 = '/Users/jdchoi/Git/character-mining-dev/json-bak'
    # input_dir2 = '/Users/jdchoi/Downloads/Friends_newly_compiled'
    # output_dir = '/Users/jdchoi/Git/character-mining/json'
    # merge_rc(input_dir1, input_dir2, output_dir)

