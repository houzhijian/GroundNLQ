## How to extract visual and textual feature 

### Textual Feature

Please refer to the [CLIP_TEXT_TOKEN_EXTRACTOR](https://github.com/houzhijian/CONE/blob/main/feature_extraction/ego4d_clip_token_extractor.py) in details.

### Video Feature

Download the EgoVLP and InternVideo features
```
python download_features.py --feature_types internvideo
```

Convert the feature into lmdb 
```
python convert_pt_to_lmdb.py 
```

