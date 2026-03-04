 # SpikeTrack
The official implementation for the **CVPR 2026** paper [_SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking_](https://arxiv.org/abs/2602.23963).

[[Models(GoogleDrive)](https://drive.google.com/drive/folders/1G9DhjfhmiRz_9JxxlbHbOnuYZBAmhLOG?usp=sharing)] [[Models(HuggingFace)](https://huggingface.co/facaiwawa/SpikeTrack)] [[Raw Results](https://drive.google.com/drive/folders/1G9DhjfhmiRz_9JxxlbHbOnuYZBAmhLOG?usp=sharing)] [[Training logs](https://drive.google.com/drive/folders/1G9DhjfhmiRz_9JxxlbHbOnuYZBAmhLOG?usp=sharing)] [[SFR EXCEL](https://drive.google.com/drive/folders/1G9DhjfhmiRz_9JxxlbHbOnuYZBAmhLOG?usp=sharing)]






## Install the environment
```
conda create -n spiketrack python=3.12
conda activate spiketrack
bash install.sh
```


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
        -- .......    
   ```


## Training
Download pre-trained backbone [SDTV3](https://github.com/BICLab/Spike-Driven-Transformer-V3/blob/main/SDT_V3/Classification/Model_Base/Train_Base.md) (5.1M for spiketrack-small / 19M for spiketrack-base) and put it under `$PROJECT_ROOT$/pretrained_models` .

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tracking/train.py --script spiketrack --config spiketrack_b256_t1 --save_dir . --mode multiple --nproc_per_node 8
```

Replace `--config` with the desired model config under `experiments/spiketrack`.


## Evaluation
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1G9DhjfhmiRz_9JxxlbHbOnuYZBAmhLOG?usp=sharing) 

Put the downloaded weights on `$PROJECT_ROOT$/ckpt`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py spiketrack spiketrack_b256_t3 --dataset lasot --threads 16 --num_gpus 4 --checkpoint_path ./ckpt/spiketrack_b256_t3.pth.tar  --inference_mode True
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test (online-server for evaluation)
```
python tracking/test.py spiketrack spiketrack_b256_t3 --dataset got10k_test --threads 16 --num_gpus 4 --checkpoint_path ./ckpt/spiketrack_b256_t3.pth.tar  --inference_mode True
python lib/test/utils/transform_got10k.py --tracker_name spiketrack --cfg_name spiketrack_b256_t3
```
- TrackingNet (online-server for evaluation)
```
python tracking/test.py spiketrack spiketrack_b256_t3 --dataset trackingnet --threads 16 --num_gpus 4 --checkpoint_path ./ckpt/spiketrack_b256_t3.pth.tar  --inference_mode True
python lib/test/utils/transform_trackingnet.py --tracker_name spiketrack --cfg_name spiketrack_b256_t3
```


## How to calculate Spike Firing Rate ?
for example get the avg SFR on GOT-10K:

STEP 1:
```
python tracking/test.py spiketrack spiketrack_b256_t3 --dataset got10k_test --threads 16 --num_gpus 4 --checkpoint_path ./ckpt/spiketrack_b256_t3.pth.tar  --inference_mode True --save_sfr True
```
you will get the avg SFR (json format) of each sequence in ./tracking/spiketrack_b256_t3/

STEP 2:


```
python tracking/get_avg_sfr.py
```
this script will calculate the average SFR  of all JSON files in the folder, so you can get the average SFR of the got10k_test set.

## Acknowledgments
* Thanks for the [SeqTrack](https://github.com/microsoft/VideoX/blob/master/SeqTrack/README.md) and [PyTracking](https://github.com/visionml/pytracking) library, which helps us to quickly implement our ideas. 


## Citation
If our work is useful for your research, please consider citing:
```
@misc{zhang2026spike,
      title={SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking}, 
      author={Qiuyang Zhang and Jiujun Cheng and Qichao Mao and Cong Liu and Yu Fang and Yuhong Li and Mengying Ge and Shangce Gao},
      year={2026},
      eprint={2602.23963},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.23963}, 
}
```
## Contact
If you have any question, feel free to email qyzhang@tongji.edu.cn. ^_^ 




