## Narration File Pre-processing


This section hosts the detail information to conduct narration file pre-processing for model pre-training.


First get official [Egoclip](https://drive.google.com/file/d/1-aaDu_Gi-Y2sQI_2rsI2D1zvQBJnHpXl/view?usp=sharing) file
and the annotated training split files from visual/moment/natural language queries challenges.
```
python create_pretrain_data_from_egoclip.py
python filter_missing_video.py
```

The first code aims to generate the pretrained data, and the second code aims to filter some train samples. 

