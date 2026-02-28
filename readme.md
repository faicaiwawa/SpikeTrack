 # SpikeTrack
The official implementation for the **CVPR 2026** paper [_SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking_].

[[Models](https://drive.google.com/drive/folders/1G9DhjfhmiRz_9JxxlbHbOnuYZBAmhLOG?usp=sharing)] [[Raw Results](https://drive.google.com/drive/folders/1G9DhjfhmiRz_9JxxlbHbOnuYZBAmhLOG?usp=sharing)] [[Training logs](https://drive.google.com/drive/folders/1G9DhjfhmiRz_9JxxlbHbOnuYZBAmhLOG?usp=sharing)] [[SFR EXCEL](https://drive.google.com/drive/folders/1G9DhjfhmiRz_9JxxlbHbOnuYZBAmhLOG?usp=sharing)]






## Install the environment
```
conda create -n spiketrack python=3.12
conda activate spiktrack
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



## Acknowledgments
* Thanks for the [SeqTrack] and [PyTracking] library, which helps us to quickly implement our ideas. 


## Citation
If our work is useful for your research, please consider citing:

## Contact
If you have any question, feel free to email qyzhang@tongji.edu.cn. ^_^ 




