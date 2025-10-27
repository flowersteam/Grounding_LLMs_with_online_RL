#!/bin/bash

SOURCECODE=uez56by@jeanzay:/gpfswork/rech/imi/uez56by/code/DLP/
# SOURCECODE=uez56by@jeanzay:/gpfswork/rech/imi/ucy39hi/DLP/storage/logs/llm_gtl_nbr_env_32_Flan_T5small_trained-embedding_action-heads_nbr_actions_3_shape_reward_beta_0_seed_*
# SOURCECODE=uez56by@jeanzay:/gpfswork/rech/imi/ucy39hi/DLP/storage/logs/llm_gtl_nbr_env_32_Flan_T5pico_action-heads_nbr_actions_3_shape_reward_beta_0_seed_*
# SOURCECODE=uez56by@jeanzay:/gpfsscratch/rech/imi/uez56by/DLP/models/llm_mtrl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_*

DESTINATIONCODE=/home/tcarta/DLP
# DESTINATIONCODE=/home/tcarta/DLP/storage/logs/
# DESTINATIONCODE=/home/tcarta/DLP/storage/models/


echo Transfer code ...
rsync -azvh -e "ssh -i $HOME/.ssh/id_rsa" --exclude-from=exclude_rsync_from_jeanzay.txt $SOURCECODE $DESTINATIONCODE


