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


S01_MAP = {
    'David Hasselhof': 'David',
    'Angela Delvecchio': 'Angela Delveccio',
    'Barry': 'Barry Farber',
    'Carl': "Carl (Rachel's date)",
    'Carol': 'Carol Willick',
    "Frannie": "Franny",
    'Janice': 'Janice Litman Goralnik',
    'Jill': 'Jill Goodacre',
    'Lizzie': 'Lizzy',
    'Luisa': 'Luisa Gianetti',
    'Mindy': 'Mindy Hunter',
    'Nurse': 'Nurse Sizemore',
    'Paul': 'Paul the Wine Guy',
    'Ronni Rappelano': 'Ronni Rapalono',
    'Steve': 'Steve (drug addict)',
}

S02_MAP = {
    'Richard': 'Richard Burke',
    'Eddie': 'Eddie Menuek',
    'Susie': 'Susie Moss',
    'Mr. Green': 'Leonard Green',
    'Van Damme': 'Jean-Claude Van Damme',
    'Stephanie': 'Stephanie Schiffer',
    'Estelle': 'Estelle Leonard',
    'Janice': 'Janice Litman Goralnik',
    'Steve': 'Steven Fisher',
    'Lipson': 'Dean Lipson',
    'Ben': 'Ben Geller',
    'Mindy': 'Mindy Hunter',
    'Carol': 'Carol Willick',
    'Susan': 'Susan Bunch',
    'Mr. Boyle': 'Buddy Boyles',
    'Barry': 'Barry Farber',
}

S03_MAP = {
    'Pete': 'Peter Becker',
    'Peter Bekcer': 'Peter Bekcer',
    'Janice': 'Janice Litman Goralnik',
    'Kate': 'Kate Miller',
    'Mark': 'Mark Robinson',
    'Richard': 'Richard Burke',
    'Dr. Green': 'Leonard Green',
    'Robert': 'Robert Bobby',
    'Phoebe Sr.': 'Phoebe Abbott',
    'Julio': 'Julio (poet)',
    'Alice': 'Alice Knight',
    'Sarah': 'Sarah Tuttle',
    'Eric': 'Eric (photographer)',
    'Whitfield': 'Sherman Whitfield',
    'Susan': 'Susan Bunch',
    'Ben': 'Ben Geller',
    'Cookie': 'Cookie Tribbiani',
    'Estelle': 'Estelle Leonard',
    'Stevenson': 'Parker Stevenson',
    'Michelle': 'Michelle Burke',
    'Carol': 'Carol Willick',
    'Johnson': 'Dr. Johnson'
}

S04_MAP = {
    'Emily': 'Emily Waltham',
    'Joshua': 'Joshua Burgin',
    'Phoebe Sr.': 'Phoebe Abbott',
    'Tim': 'Timothy Burke',
    'Alice': 'Alice Knight',
    'Janice': 'Janice Litman Goralnik',
    'Chip': 'Chip Matthews',
    'Rick': 'Rick Sanoven',
    'Ursula': 'Ursula Buffay',
    'Amanda': "Amanda (Ross' date)",
    'Susan': 'Susan Bunch',
    'Dr. Timothy Burke': 'Timothy Burke',
    'Mrs. Waltham': 'Andrea Waltham',
    'Mr. Waltham': 'Stephen Waltham',
}

S05_MAP = {
    'Steve': 'Steve Cera',
    'Emily': 'Emily Waltham',
    'Janice': 'Janice Litman Goralnik',
    'Ursula': 'Ursula Buffay',
    'Mrs. Waltham': 'Andrea Waltham',
    'Mr. Waltham': 'Stephen Waltham',
    'Alice': 'Alice Knight',
    'Estelle': 'Estelle Leonard',
    'Ben': 'Ben Geller',
}

S06_MAP = {
    'Paul': 'Paul Stevens',
    'Janine': 'Janine Lecroix',
    'Elizabeth': 'Elizabeth Stevens',
    'Jill': 'Jill Green',
    'Richard': 'Richard Burke',
    'Dana': 'Dana Keystone',
    'Estelle': 'Estelle Leonard',
    'Ursula': 'Ursula Buffay',
    'Susan': 'Susan Bunch',
    'Carl': "Carl (Joey's lookalike)",
    'Ben': 'Ben Geller',
    'Janice': 'Janice Litman Goralnik',
}

S07_MAP = {
    'Tag': 'Tag Jones',
    'Ben': 'Ben Geller',
    'Melissa': 'Melissa Warburton',
    'Richard': 'Richard Burke',
    'Kristen': 'Kristen Leigh',
    'Janine': 'Janine Lecroix',
    'Cassie': 'Cassie Geller',
    'Megan': 'Megan Bailey',
    'Ursula': 'Ursula Buffay',
    'Morse': 'Ned Morse',
    'Mrs. Bing': 'Nora Tyler Bing',
    'Mr. Bing': 'Charles Bing',
    'Estelle': 'Estelle Leonard',
    'Julie': 'Julie Graff',
    'Frannie': 'Franny',
}

S08_MAP = {
    'Will': 'Will Colbert',
    'Dr. Green': 'Leonard Green',
    'Janice': 'Janice Litman Goralnik',
    'Clifford': 'Clifford Burnett',
    'Ursula': 'Ursula Buffay',
    'Tag': 'Tag Jones',
    'Bob': "Bob (Chandler's coworker)",
    'Bobby': 'Bobby Corso',
    'Katie': 'Katie (saleswoman)',
    'Julie': 'Julie Coreger',
    'Marc': 'Marc Coreger',
    'Ben': 'Ben Geller',
    'Estelle': 'Estelle Leonard',
    'Sid': 'Sid Goralnik',
}

S09_MAP = {
    'Mike': 'Mike Hannigan',
    'Charlie': 'Charlie Wheeler',
    'Gavin': 'Gavin Mitchell',
    'Amy': 'Amy Green',
    'Bitsy': 'Bitsy Hannigan',
    'Janice': 'Janice Litman Goralnik',
    'Lowell': 'Lowell (mugger)',
    'Mugger': 'Lowell (mugger)',
    'Ben': 'Ben Geller',
    'Mr. Oberblau': 'Jarvis Oberblau',
    'Ms. Geller': 'Judy Geller',
}

S10_MAP = {
    'Mike': 'Mike Hannigan',
    'Amy': 'Amy Green',
    'Charlie': 'Charlie Wheeler',
    'Benjamin': 'Benjamin Hobart',
    'Amanda': 'Amanda Buffamonteezi',
    'Janice': 'Janice Litman Goralnik',
    'Missy': 'Missy Goldberg',
    'Mark': 'Mark Robinson',
    'Dr. Green': 'Leonard Green',
    'Estelle': 'Estelle Leonard',
    'R Zelner': 'Mr. Zelner'
}

def entity_stats(json_file, SPEAKER_MAP):
    speaker_list = []
    entity_list = []

    season = json.load(open(json_file))
    for episode in season[EPISODES]:
        scenes = episode[SCENES]
        for scene in scenes:
            for utterance in scene[UTTERANCES]:
                speakers = utterance[SPEAKERS]
                for i, speaker in enumerate(speakers):
                    speakers[i] = SPEAKER_MAP.get(speaker, speaker)
                speaker_list.extend(speakers)

                # for character_entities in utterance['character_entities']:
                #     for entities in character_entities:
                #         for i, e in enumerate(entities[2:], 2):
                #             entities[i] = SPEAKER_MAP.get(e, e)
                #         entity_list.extend(entities[2:])

    with open(json_file+'.v2','w') as fout:
        json.dump(season, fout, sort_keys=True, indent=4)

    # print('===== Entities =====')
    # c = Counter(entity_list)
    # for k, v in sorted(c.items(), key=lambda x: x[1], reverse=True):
    #     print(k+'\t'+str(v))

    print('===== Speakers =====')
    c = Counter(speaker_list)
    for k, v in sorted(c.items(), key=lambda x: x[1], reverse=True):
        print(k + '\t' + str(v))


def find(json_file):
    season = json.load(open(json_file))
    for episode in season[EPISODES]:
        scenes = episode[SCENES]
        for scene in scenes:
            for alloquies in scene['alloquies']:
                discourse = alloquies['discourseWithoutDescription']
                speakers = alloquies['speakers']
                if '' in speakers:
                    print(alloquies['alloquyId'])
                for character_entities in discourse['characterEntities']:
                    for entity_list in character_entities:
                        pass


def entity_stats(json_dir):
    g_speaker_list = []
    g_entity_list = []

    for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
        speaker_list = []
        entity_list = []
        num_mentions = 0

        season = json.load(open(json_file))
        for episode in season[EPISODES]:
            scenes = episode[SCENES]
            for scene in scenes:
                for utterance in scene[UTTERANCES]:
                    speakers = utterance[SPEAKERS]
                    # for i, speaker in enumerate(speakers):
                    #     if speaker == 'Boys' or speaker == '': print(utterance[UTTERANCE_ID])
                    speaker_list.extend(speakers)

                    if CHARACTER_ENTITIES in utterance:
                        for character_entities in utterance[CHARACTER_ENTITIES]:
                            num_mentions += len(character_entities)
                            for entities in character_entities:
                                for i, e in enumerate(entities[2:], 2):
                                    entities[i] = SM.get(e, e)
                                    # if e == 'Peter': print(utterance[UTTERANCE_ID])
                                entity_list.extend(entities[2:])

        # ordered_print(json_file, season)

        g_speaker_list.extend(speaker_list)
        g_entity_list.extend(entity_list)
        s = '\t'.join(map(str, [season[SEASON_ID], len(set(speaker_list)), num_mentions, len(set(entity_list))]))
        print(s)

    # print('===== Speakers =====')
    # c = Counter(g_speaker_list)
    # for k, v in sorted(c.items()): print(k + '\t' + str(v))
    #
    # print('===== Entities =====')
    # c = Counter(g_entity_list)
    # for k, v in sorted(c.items()): print(k+'\t'+str(v))




def get_tokens(json_dir):
    tokens = {}

    for i, json_file in enumerate(sorted(glob.glob(os.path.join(json_dir, '*.json')))):
        if i >= 4: break
        season = json.load(open(json_file))
        for episode in season[EPISODES]:
            scenes = episode[SCENES]
            for scene in scenes:
                for utterance in scene[UTTERANCES]:
                    utterance_id = utterance['utterance_id']
                    tokens[utterance_id] = utterance['tokens']

    return tokens


def compare():
    main_dir = '/Users/jdchoi/Git/character-mining-dev/json'
    ethan_dir = '/Users/jdchoi/Downloads/enhanced-jsons'

    m_utterances = get_tokens(main_dir)
    e_utterances = get_tokens(ethan_dir)
    c = 0
    for utterance_id, m_tokens in m_utterances.items():
        e_tokens = e_utterances[utterance_id]
        e_tokens = [tokens for tokens in e_tokens if len(tokens) > 1 or tokens[0] != '_']

        if len(m_tokens) != len(e_tokens):
            print(utterance_id)
            print(m_tokens)
            print(e_tokens)
            c += 1

        for tokens in e_tokens:
            if 'hes' in tokens:
                print(tokens)

    print(c)


if __name__ == '__main__':
    pass