import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_log(dir_):
    """Loads log from a directory and adds it to a list of dataframes."""
    df = pd.read_csv(os.path.join(dir_, 'log.csv'),
                     on_bad_lines='warn')
    if not len(df):
        print("empty df at {}".format(dir_))
        return
    df['model'] = dir_
    return df


def load_logs(root):
    dfs = []
    for root, dirs, files in os.walk(root, followlinks=True):
        for file_ in files:
            if file_ == 'log.csv':
                dfs.append(load_log(root))
    return dfs


def plot_average_impl(df, regexps, labels, limits, colors, y_value='return_mean', window=10, agg='mean',
                      x_value='frames'):
    """Plot averages over groups of runs  defined by regular expressions."""
    df = df.dropna(subset=[y_value])
    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                    for regex in regexps]

    for regex, models, label, color in zip(regexps, model_groups, labels, colors):
        # print("regex: {}".format(regex))
        print(models)
        df_re = df[df['model'].isin(models)]
        # the average doesn't make sense if most models are not included,
        # so we only for the period of training that has been done by all models
        num_frames_per_model = [df_model[x_value].max()
                                for _, df_model in df_re.groupby('model')]
        for _, df_model in df_re.groupby('model'):
            print(df_model[x_value].max())
        median_progress = sorted(num_frames_per_model)[(len(num_frames_per_model) - 1) // 2]
        mean_duration = np.mean([
            df_model['duration'].max() for _, df_model in df_re.groupby('model')])
        df_re = df_re[df_re[x_value] <= limits]

        # smooth
        parts = []
        for _, df_model in df_re.groupby('model'):
            df_model = df_model.copy()
            df_model.loc[:, y_value] = df_model[y_value].rolling(window).mean()
            parts.append(df_model)
        df_re = pd.concat(parts)
        df_agg = df_re.groupby([x_value]).mean()
        # df_max = df_re.groupby([x_value]).max()[y_value]
        # df_min = df_re.groupby([x_value]).min()[y_value]
        values = df_agg[y_value]
        std = df_re.groupby([x_value]).std()[y_value]
        # print(std.iloc[-1])
        df_max = values + std
        df_min = values - std

        # pyplot.plot(df_agg.index, values, label='{} SE: {}'.format(label, round(values.sum()/len(values), 3)))
        print(("{} last mean:{} last std: {}").format(label, values.iloc[-1], std.iloc[-1]))
        plt.plot(df_agg.index, values, label=label, color=color)
        # pyplot.plot(df_agg.index, values, label=label)
        plt.fill_between(df_agg.index, df_max, df_min, alpha=0.25, color=color)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])
        print("{} sample efficiency: {}".format(label, values.sum() / len(values)))


def plot_average_impl_ax(df, regexps, ax, labels, limits, colors, y_value='return_mean', window=10, agg='mean',
                      x_value='frames'):
    """Plot averages over groups of runs  defined by regular expressions."""
    df = df.dropna(subset=[y_value])
    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                    for regex in regexps]

    for regex, models, label, color in zip(regexps, model_groups, labels, colors):
        # print("regex: {}".format(regex))
        print(models)
        df_re = df[df['model'].isin(models)]
        # the average doesn't make sense if most models are not included,
        # so we only for the period of training that has been done by all models
        num_frames_per_model = [df_model[x_value].max()
                                for _, df_model in df_re.groupby('model')]
        for _, df_model in df_re.groupby('model'):
            print(df_model[x_value].max())
        median_progress = sorted(num_frames_per_model)[(len(num_frames_per_model) - 1) // 2]
        mean_duration = np.mean([
            df_model['duration'].max() for _, df_model in df_re.groupby('model')])
        df_re = df_re[df_re[x_value] <= limits]

        # smooth
        parts = []
        for _, df_model in df_re.groupby('model'):
            df_model = df_model.copy()
            df_model.loc[:, y_value] = df_model[y_value].rolling(window).mean()
            parts.append(df_model)
        df_re = pd.concat(parts)
        df_agg = df_re.groupby([x_value]).mean()
        # df_max = df_re.groupby([x_value]).max()[y_value]
        # df_min = df_re.groupby([x_value]).min()[y_value]
        values = df_agg[y_value]
        std = df_re.groupby([x_value]).std()[y_value]
        # print(std.iloc[-1])
        df_max = values + std
        df_min = values - std

        # pyplot.plot(df_agg.index, values, label='{} SE: {}'.format(label, round(values.sum()/len(values), 3)))
        print(("{} last mean:{} last std: {}").format(label, values.iloc[-1], std.iloc[-1]))
        ax.plot(df_agg.index, values, label=label, color=color)
        # pyplot.plot(df_agg.index, values, label=label)
        ax.fill_between(df_agg.index, df_max, df_min, alpha=0.25, color=color)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])
        print("{} sample efficiency: {}".format(label, values.sum() / len(values)))


dfs = load_logs('/home/tcarta/DLP/storage')
df = pd.concat(dfs, sort=True)


def plot_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='return_mean', *args, **kwargs)
    # plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11}, bbox_to_anchor=(1.1, 1.1))
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Reward", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    # plt.figure(figsize=(8, 6), dpi=100)
    plt.show()


def plot_success_rate_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='success_rate', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Success Rate", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.show()


def plot_entropy_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='entropy', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Entropy", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_loss_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='loss', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Loss", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_policy_loss_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='policy_loss', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Policy Loss", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_value_loss_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='value_loss', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Value Loss", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

def plot_grad_norm_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='grad_norm', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Grad Norm", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

# #######################MTRL############################################################## #
regexs = ['.*llm_mtrl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_mtrl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*drrn_mtrl_nbr_env_32_DRRN_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*MTRL-nbr_actions-6-PPO-NoPre.*']
labels = ['GFLAN-T5-large', 'NPAE-FLAN-T5', 'DRRN', 'Symbolic-PPO']
# limits = 3500000
limits = 1500000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:grey']
# plot_average(df, regexs, labels, limits, colors)
plot_success_rate_average(df, regexs, labels, limits, colors)

regexs = ['.*llm_mtrl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*']
labels = ['FLAN-T5-large']
# limits = 3500000
limits = 1000000
colors = ['tab:blue']
# plot_entropy_average(df, regexs, labels, limits, colors)
# plot_loss_average(df, regexs, labels, limits, colors)
# plot_policy_loss_average(df, regexs, labels, limits, colors)
# plot_value_loss_average(df, regexs, labels, limits, colors)

# ####################### Performance function of the input type ######################## #
# ####################### GoToRedBall env ######################## #

regexs = ['.*drrn_gtrb_nbr_env_32_DRRN_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*GTRB-nbr_actions-6-PPO-NoPre.*']
labels = ['DRRN_3 actions', 'Symbolic-PPO_GTRB']
limits = 400000
colors = ['tab:blue', 'tab:orange']
# plot_average(df, regexs, labels, limits, colors)
# plot_success_rate_average(df, regexs, labels, limits, colors)

# ####################### Performance function of the pretrainning ######################## #
# ####################### GoToLocation env ######################## #
# ####################### LLM_large ######################## #

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_False_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*']

# labels = ['Full pretrained & scoring', 'Full pretrained & action head', 'Not pretrained & action head', 'Pretrained embedding & scoring', 'Pretrained embedding & action head']

labels = ['GFLAN-T5', 'AFLAN-T5', 'NPAE-FLAN-T5', 'NPA-FLAN-T5', 'NPE-FLAN-T5']
limits = 500000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:grey']
# plot_average(df, regexs, labels, limits, colors)
plot_success_rate_average(df, regexs, labels, limits, colors)
# plot_entropy_average(df, regexs, labels, limits, colors)
# plot_value_loss_average(df, regexs, labels, limits, colors)

# ####################### Performance function of the size of the prompt ######################## #
# ####################### GoToLocation env ######################## #
# ####################### LLM_large ######################## #
regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_obs_1_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_obs_6_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_obs_9_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*']

# labels = ['Full pretrained & scoring', 'Full pretrained & action head', 'Not pretrained & action head', 'Pretrained embedding & scoring', 'Pretrained embedding & action head']

labels = ['1 observation', '3 observations', '6 observations', '9 observations']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
# plot_average(df, regexs, labels, limits, colors)
plot_success_rate_average(df, regexs, labels, limits, colors)

# ####################### Performance function of the size of the LLM ######################## #
# ####################### GoToLocation env ######################## #

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5xl_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5small_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*drrn_gtl_nbr_env_32_DRRN_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*GTL-nbr_actions-6-PPO-NoPre.*']
labels = ['GFLAN-T5-xl', 'GFLAN-T5-large', 'GFLAN-T5-small', 'NPAE-FLAN-T5-large', 'DRRN', 'Symbolic-PPO']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:grey', 'tab:pink']
# plot_average(df, regexs, labels, limits, colors)
# plot_success_rate_average(df, regexs, labels, limits, colors)

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5xl_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5small_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*']
labels = ['FLAN-T5-xl', 'FLAN-T5-large', 'FLAN-T5-small']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green']
# plot_entropy_average(df, regexs, labels, limits, colors)
# plot_loss_average(df, regexs, labels, limits, colors)
# plot_policy_loss_average(df, regexs, labels, limits, colors)
# plot_value_loss_average(df, regexs, labels, limits, colors)
# plot_grad_norm_average(df, regexs, labels, limits, colors)

# ####################### Performance function of the number of actions ######################## #
# ####################### GoToLocation env ######################## #
# ####################### LLM_large ######################## #

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12.5, 10))

# ####################### 3 actions ######################## #

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_3_turn_left_turn_right_go_forward_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_3_turn_left_turn_right_go_forward_shape_reward_beta_0.*',
          '.*drrn_gtl_nbr_env_32_DRRN_pretrained_True_nbr_actions_3_turn_left_turn_right_go_forward_shape_reward_beta_0.*',
          '.*GTL-nbr_actions-3-PPO-NoPre.*']
labels = ['GFLAN-T5-large', 'NPAE-FLAN-T5-large', 'DRRN', 'Symbolic-PPO']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

plot_average_impl_ax(df, regexs, ax0, labels, limits, colors, y_value='success_rate')
ax0.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
ax0.set_xlabel("Frames", fontsize=15)

ax0.set_title("Restricted", fontsize=15)
ax0.set_xticks(np.arange(stop=400001, step=50000), fontsize=10)
ax0.set_yticks(np.arange(start=0.2, stop=1, step=0.1), fontsize=10)
ax0.grid()

# ####################### 6 actions ######################## #

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*drrn_gtl_nbr_env_32_DRRN_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*GTL-nbr_actions-6-PPO-NoPre.*']
labels = ['GFLAN-T5-large', 'NPAE-FLAN-T5-large', 'DRRN', 'Symbolic-PPO']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

plot_average_impl_ax(df, regexs, ax1, labels, limits, colors, y_value='success_rate')
ax1.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
ax1.set_xlabel("Frames", fontsize=15)

ax1.set_title("Canonical", fontsize=15)
ax1.set_xticks(np.arange(stop=400001, step=50000), fontsize=10)
ax1.set_yticks(np.arange(start=0.2, stop=1, step=0.1), fontsize=10)
ax1.grid()

# ####################### 9 actions ######################## #

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_9_turn_left_turn_right_go_forward_pick_up_drop_toggle_sleep_do_nothing_think_shape_reward_beta_0*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_9_turn_left_turn_right_go_forward_pick_up_drop_toggle_sleep_do_nothing_think_shape_reward_beta_0.*',
          '.*drrn_gtl_nbr_env_32_DRRN_pretrained_True_nbr_actions_9_turn_left_turn_right_go_forward_pick_up_drop_toggle_sleep_do_nothing_think_shape_reward_beta_0.*',
          '.*GTL-nbr_actions-9-PPO-NoPre.*']
labels = ['GFLAN-T5-large', 'NPAE-FLAN-T5-large', 'DRRN', 'Symbolic-PPO']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

plot_average_impl_ax(df, regexs, ax2, labels, limits, colors, y_value='success_rate')
ax2.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
ax2.set_xlabel("Frames", fontsize=15)

ax2.set_title("Augmented", fontsize=15)
ax2.set_xticks(np.arange(stop=400001, step=50000), fontsize=10)
ax2.set_yticks(np.arange(start=0.2, stop=1, step=0.1), fontsize=10)
ax2.grid()

# ####################### LLM_mixt ######################## #

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_3_turn_left_turn_right_go_forward_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_9_turn_left_turn_right_go_forward_pick_up_drop_toggle_sleep_do_nothing_think_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_3_turn_left_turn_right_go_forward_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_9_turn_left_turn_right_go_forward_pick_up_drop_toggle_sleep_do_nothing_think_shape_reward_beta_0.*']
labels = ['GFLAN-T5-large 3 actions', 'GFLAN-T5-large 6 actions', 'GFLAN-T5-large 9 actions',
          'NPAE-FLAN-T5-large 3 actions', 'NPAE-FLAN-T5-large 6 actions', 'NPAE-FLAN-T5-large 9 actions']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:grey', 'tab:purple', 'tab:pink']

plot_average_impl_ax(df, regexs, ax3, labels, limits, colors, y_value='success_rate')
ax3.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
ax3.set_xlabel("Frames", fontsize=15)

ax3.set_title("Comparison of the 3 action spaces", fontsize=15)
ax3.set_xticks(np.arange(stop=400001, step=50000), fontsize=10)
ax3.set_yticks(np.arange(start=0.2, stop=1, step=0.1), fontsize=10)
ax3.grid()

fig.suptitle('Average Success Rate', fontsize=15)
fig.tight_layout()
plt.show()

# ####################### Performance function of the number of distractors ######################## #
# ####################### GoToLocation env ######################## #
# ####################### LLM_large ######################## #

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12.5, 10))

# ####################### 4 distractors ######################## #
regexs = ['.*llm_gtl_distractor_4_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_distractor_4_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*drrn_gtl_distractor_4_nbr_env_32_DRRN_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*GTL4-nbr_actions-6-PPO-NoPre.*']

labels = ['GFLAN-T5-large', 'NPAE-FLAN-T5-large', 'DRRN', 'Symbolic-PPO']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

plot_average_impl_ax(df, regexs, ax0, labels, limits, colors, y_value='success_rate')
ax0.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
ax0.set_xlabel("Frames", fontsize=15)

ax0.set_title("4 distractors", fontsize=15)
ax0.set_xticks(np.arange(stop=400001, step=50000), fontsize=10)
ax0.set_yticks(np.arange(start=0.2, stop=1, step=0.1), fontsize=10)
ax0.grid()


# ####################### 8 distractors ######################## #
regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*drrn_gtl_nbr_env_32_DRRN_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*GTL-nbr_actions-6-PPO-NoPre.*']

labels = ['GFLAN-T5-large', 'NPAE-FLAN-T5-large', 'DRRN', 'Symbolic-PPO']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

plot_average_impl_ax(df, regexs, ax1, labels, limits, colors, y_value='success_rate')
ax1.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
ax1.set_xlabel("Frames", fontsize=15)

ax1.set_title("8 distractors", fontsize=15)
ax1.set_xticks(np.arange(stop=400001, step=50000), fontsize=10)
ax1.set_yticks(np.arange(start=0.2, stop=1, step=0.1), fontsize=10)
ax1.grid()

# ####################### 16 distractors ######################## #
regexs = ['.*llm_gtl_distractor_16_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_distractor_16_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*drrn_gtl_distractor_16_nbr_env_32_DRRN_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*GTL16-nbr_actions-6-PPO-NoPre.*']

labels = ['GFLAN-T5-large', 'NPAE-FLAN-T5-large', 'DRRN', 'Symbolic-PPO']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

plot_average_impl_ax(df, regexs, ax2, labels, limits, colors, y_value='success_rate')
ax2.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
ax2.set_xlabel("Frames", fontsize=15)

ax2.set_title("16 distractors", fontsize=15)
ax2.set_xticks(np.arange(stop=400001, step=50000), fontsize=10)
ax2.set_yticks(np.arange(start=0.2, stop=1, step=0.1), fontsize=10)
ax2.grid()

# ####################### Mixt ######################## #
regexs = ['.*llm_gtl_distractor_4_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_distractor_16_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_distractor_4_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*',
          '.*llm_gtl_distractor_16_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*']

labels = ['GFLAN-T5-large 4 distractors', 'GFLAN-T5-large 8 distractors', 'GFLAN-T5-large 16 distractors',
          'NPAE-FLAN-T5-large 4 distractors', 'NPAE-FLAN-T5-large 8 distractors', 'NPAE-FLAN-T5-large 16 distractors']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:grey', 'tab:green', 'tab:purple', 'tab:pink']

plot_average_impl_ax(df, regexs, ax3, labels, limits, colors, y_value='success_rate')
ax3.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
ax3.set_xlabel("Frames", fontsize=15)

ax3.set_title("Comparison for 3 number of distractors", fontsize=15)
ax3.set_xticks(np.arange(stop=400001, step=50000), fontsize=10)
ax3.set_yticks(np.arange(start=0.2, stop=1, step=0.1), fontsize=10)
ax3.grid()

fig.suptitle('Average Success Rate', fontsize=15)
fig.tight_layout()
plt.show()

# ####################### Distribution shift study 6 actions ######################## #
# ############################## MixtTrainLocal ##################################### #
# ####################### LLM_large ######################## #

name_file = ['llm_mtrl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_1',
'llm_mtrl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_2']

nbr_test_prompts = 11
actions = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]

fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(15.6, 16))

columns_names = ['{}'.format(i) for i in range(len(actions)*nbr_test_prompts)]

distrib_large_1 = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[0]+"/distrib.csv", names=columns_names)
distrib_large_2 = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[1]+"/distrib.csv", names=columns_names)


len_data_frame = min(distrib_large_1.shape[0], distrib_large_2.shape[0])

for j in range(nbr_test_prompts):
    for i in range(len(actions)):
        distrib_large_action_i = 0.5*(distrib_large_1.iloc[:len_data_frame, i+6*j].values +distrib_large_2.iloc[:len_data_frame, i+6*j].values )
        ax[j//3][j % 3].plot(np.arange(len_data_frame), distrib_large_action_i, label=actions[i])
    ax[j//3][j % 3].legend()
    ax[j//3][j % 3].set_xlabel("updates")
    ax[j//3][j % 3].set_ylabel("probability")
    ax[j//3][j % 3].set_title('Prompt {}'.format(j))

fig.suptitle('Policy evolution', y=0.995, fontsize=15)
fig.tight_layout()
plt.show()

# ####################### Distribution shift study 6 actions ######################## #
# ############################## GoToLocal ##################################### #

# ####################### LLM_small ######################## #

"""name_file = ['llm_gtl_nbr_env_32_Flan_T5small_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_1',
'llm_gtl_nbr_env_32_Flan_T5small_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_2']



actions = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]
columns_names = ['{}'.format(i) for i in range(len(actions)*2)] # 6 actions 2 test prompts

distrib_large_1 = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[0]+"/distrib.csv", names=columns_names)
distrib_large_2 = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[1]+"/distrib.csv", names=columns_names)


len_data_frame = min(distrib_large_1.shape[0], distrib_large_2.shape[0])

for j in range(2):
    for i in range(len(actions)):
        distrib_large_action_i = 0.5*(distrib_large_1.iloc[:len_data_frame, i+len(actions)*j].values +distrib_large_2.iloc[:len_data_frame, i+len(actions)*j].values )
        plt.plot(np.arange(len_data_frame), distrib_large_action_i, label=actions[i])
        plt.legend()
        plt.xlabel("updates")
        plt.ylabel("probability")
        plt.title('Prompt {}'.format(j))
    plt.show()"""

# ####################### LLM_large ######################## #

"""name_file = ['llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_1',
'llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_2']



actions = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]
columns_names = ['{}'.format(i) for i in range(len(actions)*2)] # 6 actions 2 test prompts

distrib_large_1 = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[0]+"/distrib.csv", names=columns_names)
distrib_large_2 = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[1]+"/distrib.csv", names=columns_names)


len_data_frame = min(distrib_large_1.shape[0], distrib_large_2.shape[0])

for j in range(2):
    for i in range(len(actions)):
        distrib_large_action_i = 0.5*(distrib_large_1.iloc[:len_data_frame, i+len(actions)*j].values +distrib_large_2.iloc[:len_data_frame, i+len(actions)*j].values )
        plt.plot(np.arange(len_data_frame), distrib_large_action_i, label=actions[i])
        plt.legend()
        plt.xlabel("updates")
        plt.ylabel("probability")
        plt.title('Prompt {}'.format(j))
    plt.show()"""


# ####################### Distribution shift study 9 actions ######################## #
# ############################## GoToLocal ##################################### #


"""name_file = ['llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_9_turn_left_turn_right_go_forward_pick_up_drop_toggle_sleep_do_nothing_think_shape_reward_beta_0_seed_1',
'llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_9_turn_left_turn_right_go_forward_pick_up_drop_toggle_sleep_do_nothing_think_shape_reward_beta_0_seed_2']

nbr_test_prompts = 2
actions = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle", "sleep", "do_nothing", "think"]
columns_names = ['{}'.format(i) for i in range(len(actions)*nbr_test_prompts)]

distrib_large_1 = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[0]+"/distrib.csv", names=columns_names)
distrib_large_2 = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[1]+"/distrib.csv", names=columns_names)


len_data_frame = min(distrib_large_1.shape[0], distrib_large_2.shape[0])

for j in range(nbr_test_prompts):
    for i in range(len(actions)):
        distrib_large_action_i = 0.5*(distrib_large_1.iloc[:len_data_frame, i+6*j].values +distrib_large_2.iloc[:len_data_frame, i+6*j].values )
        plt.plot(np.arange(len_data_frame), distrib_large_action_i, label=actions[i])
        plt.legend()
        plt.xlabel("updates")
        plt.ylabel("probability")
        plt.title('Prompt {}'.format(j))
    plt.show()"""

# ======================================================================================================================
# ======================================================REMOVED=========================================================
# ======================================================================================================================

# ####################### Performance function of the number of rooms ######################## #
# ####################### LLM_large ######################## #
# ####################### 1 room ######################## #
regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTL-PPO-NoPre.*']
labels = ['FLAN-T5-large 1 room', 'Classic A2C 1 room']
limits = 200000
colors = ['tab:blue', 'tab:orange']
# plot_average(df, regexs, labels, limits, colors)

# ####################### 2 rooms ######################## #
regexs = ['.*llm_gtm_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTM-PPO-NoPre.*']
labels = ['FLAN-T5-large 2 rooms', 'Classic A2C 2 rooms']
limits = 200000
colors = ['tab:blue', 'tab:orange']
# plot_average(df, regexs, labels, limits, colors)

# ####################### 4 rooms ######################## #
regexs = ['.*llm_gtlarge_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTLarge-PPO-NoPre.*']
labels = ['FLAN-T5-large 4 rooms', 'Classic A2C 4 rooms']
limits = 400000
colors = ['tab:blue', 'tab:orange']
# plot_average(df, regexs, labels, limits, colors)

# ####################### Mixt ######################## #
regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtm_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtlarge_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*']
labels = ['FLAN-T5-large 1 room', 'FLAN-T5-large 2 rooms', 'FLAN-T5-large 4 rooms']
limits = 200000
colors = ['tab:blue', 'tab:orange', 'tab:grey']
# plot_average(df, regexs, labels, limits, colors)

# ####################### Performance function of the type of reward ######################## #
# ####################### LLM_large ######################## #
regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_simple_env_reward_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTL-PPO-NoPre.*']
labels = ['FLAN-T5-large', 'FLAN-T5-large-simple-reward', 'Classic-A2C']
limits = 200000
colors = ['tab:blue', 'tab:orange', 'tab:green']
# plot_average(df, regexs, labels, limits, colors)

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_simple_env_reward_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*']
labels = ['FLAN-T5-large', 'FLAN-T5-large-simple-reward', 'Classic-A2C']
limits = 200000
colors = ['tab:blue', 'tab:orange']
# plot_loss_average(df, regexs, labels, limits, colors)
# plot_policy_loss_average(df, regexs, labels, limits, colors)
# plot_value_loss_average(df, regexs, labels, limits, colors)
# plot_entropy_average(df, regexs, labels, limits, colors)

# ####################### Ablation: pretraining of the LLM large ######################## #
# ####################### LLM_large ######################## #
regexs = ['.*llm_gtl_simple_env_reward_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_untrained_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTL-PPO-NoPre.*']
labels = ['FLAN-T5-large-simple-reward', 'FLAN-T5-large-untrained', 'Classic-A2C']
limits = 250000
colors = ['tab:blue', 'tab:orange', 'tab:green']
# plot_average(df, regexs, labels, limits, colors)
# plot_loss_average(df, regexs, labels, limits, colors)




# ####################### Distribution shift study 3 actions ######################## #

"""name_file = ['llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed_1',
             'llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed_2',
             'llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_3_shape_reward_beta_0_seed_1',
             'llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_3_shape_reward_beta_0_seed_2']

legend = ['T5_large_1',
          'T5_large_2',
          'T5_small_1',
          'T5_small_2'
          ]

columns_names=['{}'.format(i) for i in range(3*6)]
indices = np.arange(3)
actions = ["turn left", "turn right", "go forward"]
width = 0.1
for i in range(len(name_file)):
    for j in range(4):
        distrib_large = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[i]+"/distrib.csv", names=columns_names)
        # p = plt.bar(indices-0.15+0.1*j, distrib_large.iloc[j][:3], width=width, alpha=0.5, label="update: {}".format(j*50))
        p = plt.bar(indices-0.15+0.1*j, distrib_large.iloc[j][12:15], width=width, alpha=0.5, label="update: {}".format(j*50))
        plt.xticks(indices, actions)
        plt.legend()
    plt.title(legend[i])
    plt.show()"""

# ####################### Distribution shift study 3 actions untrained ######################## #

"""name_file = ['llm_gtl_nbr_env_32_Flan_T5large_untrained_nbr_actions_3_shape_reward_beta_0_seed_1',
             'llm_gtl_nbr_env_32_Flan_T5large_untrained_nbr_actions_3_shape_reward_beta_0_seed_2']

legend = ['T5_large_untrained_1',
          'T5_untrained_large_2']

columns_names=['{}'.format(i) for i in range(3*6)]
indices = np.arange(3)
actions = ["turn left", "turn right", "go forward"]
width = 0.1
for i in range(len(name_file)):
    for j in range(4):
        distrib_large = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[i]+"/distrib.csv", names=columns_names)
        # p = plt.bar(indices-0.15+0.1*j, distrib_large.iloc[j][:3], width=width, alpha=0.5, label="update: {}".format(j*50))
        p = plt.bar(indices-0.15+0.1*j, distrib_large.iloc[j][3:], width=width, alpha=0.5, label="update: {}".format(j*50))
        plt.xticks(indices, actions)
        plt.legend()
    plt.title(legend[i])
    plt.show()"""

# ####################### Distribution shift study 9 actions ######################## #

"""name_file = ['llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_9_shape_reward_beta_0_seed_1',
             'llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_9_shape_reward_beta_0_seed_2',
             'llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_9_shape_reward_beta_0_seed_1',
             'llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_9_shape_reward_beta_0_seed_2']

legend = ['T5_small_1',
          'T5_small_2',
          'T5_large_1',
          'T5_large_2']

columns_names=['{}'.format(i) for i in range(9*6)]
indices = np.arange(9)
actions = ["turn left", "turn right", "go forward", "eat", "dance", "sleep", "do nothing", "cut", "think"]
width = 0.1
for i in range(len(name_file)):
    for j in range(2):
        distrib_large = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[i]+"/distrib.csv", names=columns_names)
        # p = plt.bar(indices-0.05+0.1*j, distrib_large.iloc[j][:9], width=width, alpha=0.5, label="update: {}".format(j*50))
        p = plt.bar(indices-0.15+0.1*j, distrib_large.iloc[j][9:], width=width, alpha=0.5, label="update: {}".format(j*50))
        plt.xticks(indices, actions, rotation=25)
        plt.legend()
    plt.title(legend[i])
    plt.show()"""