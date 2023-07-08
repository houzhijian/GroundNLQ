from basic_utils import load_json, save_jsonl
import tqdm
import pandas as pd

nlq_train_data = "../ego4d_nlq_ori_data/nlq_train.json"
moment_train_data = "../ego4d_mq_ori_data/moments_train.json"
vq_train_data = "../ego4d_vq_ori_data/vq_train.json"

metadata = pd.read_csv('egoclip.csv', sep='\t', error_bad_lines=False)
print(metadata.shape)
clip_id_list = []
final_data = []

###### get the correspoding videos
###### and egoclip timestamp boundaries for intialization
for data_filename in [nlq_train_data, moment_train_data, vq_train_data]:
    train_task_data = load_json(data_filename)
    for video_datum in tqdm.tqdm(train_task_data["videos"]):
        video_uid = video_datum['video_uid']
        video_metadata = metadata[metadata['video_uid'] == video_uid]
        if len(video_metadata) == 0:
            continue
        for clip_datum in video_datum["clips"]:
            clip_id = clip_datum['clip_uid']
            if clip_id in clip_id_list:
                continue
            clip_id_list.append(clip_id)
            temp_metadata = video_metadata[video_metadata['clip_start'] > clip_datum["video_start_sec"]]
            final_metadata = temp_metadata[temp_metadata['clip_end'] < clip_datum["video_end_sec"]]
            # ********************************
            final_metadata.loc[:, 'clip_start'] = final_metadata.loc[:, 'clip_start'] - clip_datum["video_start_sec"]
            final_metadata.loc[:, 'clip_end'] = final_metadata.loc[:, 'clip_end'] - clip_datum["video_start_sec"]
            final_metadata.loc[:, 'narration_time'] = final_metadata.loc[:, 'narration_time'] - clip_datum["video_start_sec"]
            # ********************************
            final_metadata['clip_id'] = clip_id
            final_metadata['clip_dur'] = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
            final_data.append(final_metadata)

df = pd.concat(final_data)
print(df)
print(df.shape)
pretrained_data = df.to_dict("records")

###### convert to our customized format
new_data = []
for item in tqdm.tqdm(pretrained_data):
    clip_id = item["clip_id"]
    narration_ind = item["narration_ind"]
    if item["narration_source"] == "narration_pass_1":
        middle_text = "narrator-1"
    elif item["narration_source"] == "narration_pass_2":
        middle_text = "narrator-2"
    else:
        assert 1 == 2, print(item)
    query_id = f'{clip_id}_{middle_text}_{narration_ind}'

    clip_text = item["clip_text"]
    clip_start = item["clip_start"]
    clip_end = item["clip_end"]
    clip_dur = item["clip_dur"]

    temp_dict = {
        "narration_query": clip_text,
        "query_id": query_id,
        "duration": clip_dur,
        "video_id": clip_id,
        "query_type": "nlq_narration",
        "timestamps": [[clip_start, clip_end]],
    }
    new_data.append(temp_dict)

### save the jsonl
save_jsonl(new_data, "format_unique_pretrain_data.jsonl")
