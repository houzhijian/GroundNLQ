from basic_utils import load_jsonl, save_jsonl


###remove some videos that does not have extracted video features.
video_list_id = ['1f980b47-4fe6-418f-be03-a0c23768cd74',
                 '26e62afd-05ec-450a-9291-8936f8e52571',
                 '0fa5b9ca-6ff0-4dea-8093-054dbe214f02']

filename = "./ego4d_data/narrations/format_unique_pretrain_data.jsonl"

data = load_jsonl(filename)
print("data",len(data))

new_data = [item for item in data if item["video_id"] not in video_list_id]
print("new data",len(new_data))

save_jsonl(new_data,"./ego4d_data/narrations/format_unique_pretrain_data_v2.jsonl")