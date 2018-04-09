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
CHARACTER_ENTITIES = 'character_entities'
EMOTION = 'emotion'
CAPTION = 'caption'


def ordered_print(json_file, s=None):
    def pair(key, d):
        s = d[key]
        if isinstance(s, str): s = ' '.join(s.split())
        return key, s

    if s is None: s = json.load(open(json_file))
    season = OrderedDict([pair(SEASON_ID, s), pair(EPISODES, s)])
    if len(s) != len(season): print('Error: 0')
    episodes = season[EPISODES]

    for i, e in enumerate(episodes):
        episode = OrderedDict([pair(EPISODE_ID, e), pair(SCENES, e)])
        if len(e) != len(episode): print('Error: 1')
        episodes[i] = episode
        scenes = episode[SCENES]

        for j, c in enumerate(scenes):
            scene = OrderedDict([pair(SCENE_ID, c), pair(UTTERANCES, c)])
            if len(c) != len(scene): print('Error: 2')
            scenes[j] = scene
            utterances = scene[UTTERANCES]

            for k, u in enumerate(utterances):
                utterance = [
                    pair(UTTERANCE_ID, u),
                    pair(SPEAKERS, u),
                    pair(TRANSCRIPT, u),
                    pair(TRANSCRIPT_WITH_NOTE, u),
                    pair(TOKENS, u),
                    pair(TOKENS_WITH_NOTE, u)]

                if CHARACTER_ENTITIES in u: utterance.append(pair(CHARACTER_ENTITIES, u))
                if EMOTION in u: utterance.append(pair(EMOTION, u))
                if CAPTION in u: utterance.append(pair(CAPTION, u))

                if len(u) != len(utterance): print('Error: 3')
                utterances[k] = OrderedDict(utterance)

    with open(json_file+'.v2','w') as fout:
        json.dump(season, fout, indent=4)


def general_stats(json_file):
    num_scenes = 0
    num_utterances = 0
    num_utterances_wn = 0
    num_sentences = 0
    num_sentences_wn = 0
    num_tokens = 0
    num_tokens_wn = 0
    all_speakers = set()

    season = json.load(open(json_file))
    episodes = season[EPISODES]

    for episode in episodes:
        scenes = episode[SCENES]
        num_scenes += len(scenes)

        for scene in scenes:
            utterances = scene[UTTERANCES]
            num_utterances_wn += len(utterances)

            for utterance in utterances:
                all_speakers.update(utterance[SPEAKERS])

                tokens = utterance[TOKENS]
                if tokens:
                    num_utterances += 1
                    num_sentences += len(tokens)
                    num_tokens += sum([len(t) for t in tokens])

                tokens_wn = utterance[TOKENS_WITH_NOTE] or tokens
                num_sentences_wn += len(tokens_wn)
                num_tokens_wn += sum([len(t) for t in tokens_wn])

    return [season['season_id'], len(episodes), num_scenes, num_utterances, num_sentences, num_tokens, all_speakers, num_utterances_wn, num_sentences_wn, num_tokens_wn]


def print_general_stats(json_dir):
    all_speakers = set()
    print('\t'.join(['Season ID', 'Episodes', 'Scenes', 'Utterances', 'Sentences', 'Tokens', 'Speakers']))
    for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
        l = general_stats(json_file)
        all_speakers.update(l[6])
        l[6] = len(l[6])
        print('\t'.join(map(str, l)))
    print('All speakers: %s' % (len(all_speakers)))


def entity_stats(json_dir):
    g_speaker_list = []
    g_entity_list = []

    print('\t'.join(['Season ID', 'Episodes', 'Scenes', 'Utterances', 'Tokens', 'Speakers', 'Entities', 'Singular', 'Plural', 'Mentions']))

    for k, json_file in enumerate(sorted(glob.glob(os.path.join(json_dir, '*.json')))):
        if k >= 4: break
        speaker_list = []
        entity_list = []
        num_clusters = 0
        num_scenes = 0
        num_utterances = 0
        num_tokens = 0
        num_mentions = 0
        num_singular_mentions = 0
        num_plural_mentions = 0
        entity_types = [0, 0, 0, 0, 0]

        season = json.load(open(json_file))
        episodes = season[EPISODES]

        for episode in episodes:
            scenes = episode[SCENES]

            for scene in scenes:
                annotated = False
                cluster_set = set()

                for utterance in scene[UTTERANCES]:
                    if CHARACTER_ENTITIES in utterance and len(utterance[TOKENS]) > 0:
                        annotated = True
                        num_utterances += 1
                        num_tokens += len(utterance[TOKENS])
                        speaker_list.extend(utterance[SPEAKERS])

                        for character_entities in utterance[CHARACTER_ENTITIES]:
                            # num_mentions += len(character_entities)
                            for entities in character_entities:
                                if 'Non-Entity' in entities: continue
                                for e in entities[2:]:
                                    entity_list.append(e)
                                    cluster_set.add(e)

                                    if e in {'Girl', 'Girl 1', 'Girl 2', 'Guy', 'Guy 1', 'Man', 'Man 1', 'Man 2', 'Man 3', 'Person 1', 'Person 2', 'Person 3', 'Woman', 'Woman 1', 'Woman 2', 'Woman 3'}:
                                        entity_types[2] += 1
                                    elif e in {'Monica Geller', 'Ross Geller', 'Rachel Green', 'Joey Tribbiani', 'Phoebe Buffay', 'Chandler Bing'}:
                                        entity_types[0] += 1
                                    elif e == '#GENERAL#':
                                        entity_types[3] += 1
                                    elif e == '#OTHER#':
                                        entity_types[4] += 1
                                    else:
                                        entity_types[1] += 1

                                if len(entities) == 3: num_singular_mentions += 1
                                else: num_plural_mentions += 1
                                num_mentions += 1

                if annotated: num_scenes += 1
                num_clusters += len(cluster_set)

        g_speaker_list.extend(speaker_list)
        g_entity_list.extend(entity_list)
        s = '\t'.join(map(str, [season[SEASON_ID], len(episodes), num_scenes, num_utterances, num_tokens, len(set(speaker_list)), num_singular_mentions, num_plural_mentions, num_mentions, num_clusters, len(set(entity_list))]))
        print(s)

    print('All speakers: %s' % (len(set(g_speaker_list))))
    print('All entities: %s' % (len(set(g_entity_list))))




if __name__ == '__main__':
    json_dir = '/Users/jdchoi/Git/character-mining-dev/json'
    # print_general_stats(json_dir)
    entity_stats(json_dir)
    #
    # # for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
    # #     print(json_file)
    # #     ordered_print(json_file)



