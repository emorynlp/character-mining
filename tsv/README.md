# Parsed transcripts from **.json** to **.tsv** file

## Parse all transcripts to a single file

Using the **.json** files from from each season, we create a master file that contain all transcripts in a easier to work with format.

The notebook that created **friends_transcripts.tsv** is in **/doc/json_tsv.ipynb**.

This is a sample of the **.tsv** file:

<br>

|season_id|episode_id|scene_id|utterance_id|speaker|tokens|transcript|
|:-|:-|:-|:-|:-|:-|:-|
|0|	s01|	e01|	c01|	u001|	Monica Geller|	[[There, 's, nothing, to, tell, !], [He, 's, j...|	There's nothing to tell! He's just some guy I ...|
|1|	s01|	e01|	c01|	u002|	Joey Tribbiani|	[[C'mon, ,, you, 're, going, out, with, the, g...|	C'mon, you're going out with the guy! There's ...|

<br>

## Download with one command:

```bash 
wget https://raw.githubusercontent.com/gmihaila/character-mining/developer/tsv/friends_transcripts.tsv
```

