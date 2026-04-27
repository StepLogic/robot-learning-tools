# mkdir -p /media/kojogyaase/disk_two/Research/recovery-from-failure/data/scene_datasets/gibson                                           
                                                                                                                                
# cd /media/kojogyaase/disk_two/Research/recovery-from-failure/data/scene_datasets/gibson             
# # wget https://storage.googleapis.com/gibson_scenes/assets_core_v2.tar.gz 
# # wget https://storage.googleapis.com/gibson_scenes/dataset.tar.gz 

# tar -zxf assets_core_v2.tar.gz -C core/
# tar -zxf dataset.tar.gz -C assets/
# rm assets_core_v2.tar.gz dataset.tar.gz

# python train_habitat_her.py --scene_path data/versioned_data/habitat_test_scenes/apartment_1.glb --scene_dataset_path ""
python train_habitat_her.py --scene_path /home/kojogyaase/Projects/Research/recovery-from-failure/data/gibson/Cantwell.glb --scene_dataset_path "" --video_interval 10000 --video_length 200 --debug_render