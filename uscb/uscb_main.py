import pandas as pd
import numpy as np
import datetime
import os
import random
import RL_brain_uscb as td3
import math

import torch
import torch.utils.data

import config
import logging
import sys

np.seterr(all='raise')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def bidding(bid):
    return int(bid if bid <= 300 else 300)


def generate_bid_price(datas):
    '''
    :param datas: type list
    :return:
    '''
    return np.array(list(map(bidding, datas))).astype(int)


def bid_main(bid_prices, imp_datas, budget):
    '''
    主要竞标程序
    :param bid_prices:
    :param imp_datas:
    :return:
    '''
    win_imp_indexs = np.where(bid_prices >= imp_datas[:, 2])[0]
    # 赢标数据
    win_imp_datas = imp_datas[win_imp_indexs, :]

    win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost = 0, 0, 0, 0, 0, 0, 0
    if len(win_imp_datas):
        # 二分查找
        first, last = 0, win_imp_datas.shape[0] - 1

        # 购买到的最后一个展示机会在win_imp_indexs的索引
        # 如果为0，则全买到
        final_index = 0
        while first <= last:
            mid = first + (last - first) // 2
            # 计算花费
            tmp_sum = np.sum(win_imp_datas[:mid, 2])
            # 如果花费小于预算
            if tmp_sum < budget:
                # 则前半部分的就买到
                first = mid + 1
            else:
                # 去除前半部分的最后一个
                last_sum = np.sum(win_imp_datas[:mid - 1, 2])
                # 如果花费小于预算
                if last_sum <= budget:
                    # 标记索引
                    final_index = mid - 1
                    break
                else:
                    # 又在前半部分搜索
                    last = mid - 1

        # 获胜广告预算范围内的最终索引
        # 如果全部都能买到，final_index=first
        final_index = final_index if final_index else first
        # 获胜点击
        win_clks = np.sum(win_imp_datas[:final_index, 0])
        # 获胜pctr
        win_pctr = np.sum(win_imp_datas[:final_index, 1])
        # 全部广告预算范围内的最终索引
        origin_index = win_imp_indexs[final_index - 1]
        # 真实点击
        real_clks = np.sum(imp_datas[:origin_index, 0])
        # 真实pctr
        real_pctr = np.sum(imp_datas[:origin_index, 1])
        # 获胜数
        imps = final_index
        # 真实展示数
        bids = origin_index + 1

        cost = np.sum(win_imp_datas[:final_index, 2])
        current_cost = cost

        # 如果还有能够购买的展示
        if len(win_imp_datas[final_index:, :]) > 0:
            # 如果还有预算
            if current_cost < budget:
                # 剩余预算
                budget -= current_cost

                remain_win_imps = win_imp_datas[final_index:, :]
                # 剩余预算能够购买的广告索引
                mprice_less_than_budget_imp_indexs = np.where(remain_win_imps[:, 2] <= budget)[0]

                final_mprice_lt_budget_imps = remain_win_imps[mprice_less_than_budget_imp_indexs]
                last_win_index = 0
                for idx, imp in enumerate(final_mprice_lt_budget_imps):
                    tmp_mprice = final_mprice_lt_budget_imps[idx, 2]
                    if budget - tmp_mprice >= 0:
                        # 如果能够买
                        win_clks += final_mprice_lt_budget_imps[idx, 0]
                        win_pctr += final_mprice_lt_budget_imps[idx, 1]
                        imps += 1
                        # bids += (mprice_less_than_budget_imp_indexs[idx] - last_win_index + 1)
                        last_win_index = mprice_less_than_budget_imp_indexs[idx]
                        bids = win_imp_indexs[final_index + last_win_index] + 1
                        cost += tmp_mprice
                        budget -= tmp_mprice
                    else:
                        continue
                real_clks += np.sum(remain_win_imps[:last_win_index, 0])
                real_pctr += np.sum(remain_win_imps[:last_win_index, 1])
            else:
                win_clks, real_clks, bids, imps, cost = 0, 0, 0, 0, 0
                last_win_index = 0
                for idx, imp in enumerate(win_imp_datas):
                    tmp_mprice = win_imp_datas[idx, 2]
                    real_clks += win_imp_datas[idx, 0]
                    real_pctr += win_imp_datas[idx, 1]
                    if budget - tmp_mprice >= 0:
                        win_clks += win_imp_datas[idx, 0]
                        win_pctr += win_imp_datas[idx, 1]
                        imps += 1
                        bids += (win_imp_indexs[idx] - last_win_index + 1)
                        last_win_index = win_imp_indexs[idx]
                        cost += tmp_mprice
                        budget -= tmp_mprice

    return win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost


def get_model(args, device):
    RL_model = td3.TD3_Model(args.neuron_nums,
                             args.input_dims,
                             action_nums=1,
                             lr_A=args.lr_A,
                             lr_C=args.lr_C,
                             memory_size=args.memory_size,
                             tau=args.tau,
                             batch_size=args.rl_batch_size,
                             reward_decay=0.0,
                             device=device
                             )

    return RL_model


def get_dataset(args):
    # 数据地址
    data_path = os.path.join(args.data_path + args.dataset_name, args.campaign_id)

    # 获取数据
    train_data_df = pd.read_csv(os.path.join(data_path, 'train.bid.lin.csv'))
    
    if args.test_with_train:
        test_data_df = train_data_df
    else:
        test_data_df = pd.read_csv(os.path.join(data_path, 'test.bid.lin.csv'))
    
    # 获取每日预算、ecpc、平均点击率、平均成交价
    budget = []
    for index, day in enumerate(train_data_df.day.unique()):
        current_day_budget = np.sum(train_data_df[train_data_df.day.isin([day])].market_price)

        budget.append(current_day_budget)

    return train_data_df, test_data_df, budget


def reward_func(uscb_clks, hb_clks, uscb_cost, hb_cost, uscb_pctrs):
    if uscb_clks >= hb_clks and uscb_cost < hb_cost:
        r = 5
    elif uscb_clks >= hb_clks and uscb_cost >= hb_cost:
        r = 1
    elif uscb_clks < hb_clks and uscb_cost >= hb_cost:
        r = -5
    else:
        r = -2.5

    return r / 1000



def choose_init_base_bid(config):
    base_bid_path = os.path.join('../lin/result/ipinyou/{}/normal/test'.format(config['campaign_id']),
                                 'test_bid_log.csv')
    if not os.path.exists(base_bid_path):
        raise FileNotFoundError('Run LIN first before you train uscb')
    data = pd.read_csv(base_bid_path)
    base_bid = data[data['budget_prop'] == config['budget_para']].iloc[0]['base_bid']
    avg_pctr = data[data['budget_prop'] == config['budget_para']].iloc[0]['average_pctr']

    return avg_pctr, base_bid


if __name__ == '__main__':
    # 获取参数
    args = config.init_parser()
    time_fraction = args.time_fraction
    time_fraction_str = str(time_fraction) + '_time_fraction'
    # 获取数据集
    train_data_df, test_data_df, budget = get_dataset(args)
    # 设置随机数种子
    setup_seed(args.seed)
    # 记录日志地址
    log_dirs = [args.save_log_dir, os.path.join(args.save_log_dir, args.campaign_id)]
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.save_log_dir, str(args.campaign_id),
                                              args.model_name + '_output.log'),
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    # 预测结果存放文件夹位置 uscb
    result_path = os.path.join(args.result_path, args.campaign_id)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    device = torch.device(args.device)  # 指定运行设备

    logger.info(args.campaign_id)
    logger.info('RL model ' + args.model_name + ' has been training')
    # logger.info(args)

    # 生成TD3实例
    rl_model = get_model(args, device)
    B = [b / args.budget_para for b in budget]

    avg_ctr, hb_base = choose_init_base_bid(vars(args))

    train_losses = []

    logger.info('para:{}, budget:{}, base bid: {}'.format(args.budget_para, B, hb_base))
    logger.info('\tclks\treal_clks\tbids\timps\tcost')

    start_time = datetime.datetime.now()

    # 数据索引
    clk_index, ctr_index, mprice_index, hour_index = 0, 1, 2, 3

    ep_train_records = []
    ep_test_records = []
    ep_test_actions = []
    for ep in range(args.episodes):
        train_records = [0, 0, 0, 0, 0, 0, 0]
        # win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost
        critic_loss = 0
        total_r = 0.0
        total_r2 = 0.0
        for day_index, day in enumerate(train_data_df.day.unique()):
            train_data = train_data_df[train_data_df.day.isin([day])]
            train_data = train_data[['clk', 'pctr', 'market_price', time_fraction_str]].values.astype(float)
            budget = B[day_index]
            current_day_budget = budget
            
            base_win_clks = 0
            base_budget = B[day_index]
            for t in range(time_fraction):
                if base_budget <= 0:
                    break
                hour_datas = train_data[train_data[:, hour_index] == t]
                bid_datas = generate_bid_price((hour_datas[:, ctr_index] * (hb_base / avg_ctr)))
                res_ = bid_main(bid_datas, hour_datas, base_budget)               
                base_win_clks += res_[0]
                base_budget -= res_[-1]
                
            #tmp_state = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
            #init_state = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
            tmp_state = [1, 0, 0, 0, 1]
            init_state = [1, 0, 0, 0, 1]

            done = 0
            curr_win_clks = 0
            curr_ratio = 1.0
            prev_ratio = 1.0
            for t in range(time_fraction):
                if budget > 0:
                    hour_datas = train_data[train_data[:, hour_index] == t]

                    state = torch.tensor(init_state).float() if not t else torch.tensor(tmp_state).float()

                    action = rl_model.choose_action(state.unsqueeze(0))[0, 0].item()
                    if args.cont_tune:
                        curr_ratio = prev_ratio * (1 + action)
                    else:
                        curr_ratio = 1 + action
                    bid_datas = generate_bid_price((hour_datas[:, ctr_index] * (hb_base / avg_ctr)) / curr_ratio)
                    res_ = bid_main(bid_datas, hour_datas, budget)
                    # win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost

                    train_records = [train_records[i] + res_[i] for i in range(len(train_records))]

                    left_hour_ratio = (time_fraction - t) / (time_fraction) if t <= (time_fraction - 1) else 0

                    if (not left_hour_ratio) or (budget <= 0):
                        done = 1
                    
                    curr_cpc = res_[-1] / res_[0] if res_[0] > 0 else 3 * args.tCPC
                    accu_cpc = train_records[-1] / train_records[0] if train_records[0] > 0 else 3 * args.tCPC
                    # avg_budget_ratio, cost_ratio, ctr, win_rate
                    #next_state = [(budget / current_day_budget) / left_hour_ratio if left_hour_ratio else (
                    #        budget / current_day_budget),
                    #              (budget / current_day_budget),
                    #              left_hour_ratio,
                    #              curr_cpc / args.tCPC,
                    #              accu_cpc / args.tCPC,
                    #              res_[6] / current_day_budget,
                    #              res_[0] / res_[5] if res_[5] > 0 else 1.0,
                    #              res_[0] / res_[1] if res_[1] > 0 else 1.0,
                    #              res_[5] / res_[4] if res_[4] > 0 else 1.0,
                    #              train_records[0] / train_records[5] if train_records[5] > 0 else 1.0,
                    #              train_records[0] / train_records[1] if train_records[1] > 0 else 1.0,
                    #              train_records[5] / train_records[4] if train_records[4] > 0 else 1.0                                  
                    #              ]
                    next_state = [(budget / current_day_budget) / left_hour_ratio if left_hour_ratio else (
                            budget / current_day_budget),
                                  res_[6] / current_day_budget,
                                  res_[0] / res_[5] if res_[5] else 0,
                                  res_[5] / res_[4] if res_[4] else 0,
                                  prev_ratio]                    
                    prev_ratio = curr_ratio
                    budget -= res_[-1]
                    tmp_state = next_state
                    budget_mc = budget
                    curr_win_clks += res_[0]
                    nxt_win_clks = 0
                    for t2 in range(t+1, time_fraction):
                        if budget_mc <= 0:
                            break
                        hour_datas_mc = train_data[train_data[:, hour_index] == t2]
                        bid_datas_mc = generate_bid_price((hour_datas_mc[:, ctr_index] * (hb_base / avg_ctr)) / curr_ratio)
                        res_mc = bid_main(bid_datas_mc, hour_datas_mc, budget_mc)
                        nxt_win_clks += res_mc[0]
                        budget_mc = budget_mc - res_mc[-1]
                    win_clks =  curr_win_clks + nxt_win_clks
                    r_t = win_clks / base_win_clks if base_win_clks > 0 else 3.0
                    total_r += r_t
                    cost = B[day_index] - budget_mc
                    cpc = cost / win_clks if win_clks > 0 else args.tCPC * 10
                    if args.reward_type == 0:
                        r_t2 = (args.lamda ** max(0.0, cpc / (args.tCPC * (1+args.margin)) - 1.0)  - 1)
                        total_r2 -= r_t2
                        r_t -= args.beta * r_t2
                    else:
                        ratio = 1 / (1 + math.exp(-args.lamda * (1 + args.margin - cpc / args.tCPC)))
                        #r_t2 = (args.beta * cost + win_clks) * ratio 
                        r_t2 = (args.beta * cost / B[day_index] + r_t) * ratio 
                        total_r2 += r_t2
                        r_t = r_t2
                    transitions = torch.cat([state, torch.tensor([action]).float(),
                                             torch.tensor(next_state).float(),
                                             torch.tensor([done]).float(), torch.tensor([r_t]).float()],
                                            dim=-1).unsqueeze(
                        0).to(device)

                    rl_model.store_transition(transitions)

                    if rl_model.memory.memory_counter >= args.rl_batch_size:
                        critic_loss = rl_model.learn()

        ep_train_records.append([ep] + train_records + [critic_loss, total_r, total_r2] )
        print(ep, 'train', train_records, critic_loss, total_r, total_r2)

        # win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost
        test_records = [0, 0, 0, 0, 0, 0, 0]
        # win_clks, win_pctr, cost of the days within ROI range
        test_rewards = [0, 0, 0]
        day_cpc = [] 
        day_budget = []
        test_actions = []
        
        for day_index, day in enumerate(test_data_df.day.unique()):
            current_day_test_action = [(0,0) for _ in range(time_fraction)]
            test_data = test_data_df[test_data_df.day.isin([day])]
            test_data = test_data[['clk', 'pctr', 'market_price', time_fraction_str]].values.astype(float)
            #tmp_test_state = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
            #init_test_state = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
            tmp_test_state = [1, 0, 0, 0, 1]
            init_test_state = [1, 0, 0, 0, 1]

            budget = np.sum(test_data_df[test_data_df.day.isin([day])].market_price) / args.budget_para
            current_day_budget = budget
            day_budget += [current_day_budget]
            hour_t = 0
            curr_rewards = [0, 0, 0]
            curr_ratio = 1.0
            prev_ratio = 1.0
            for t in range(time_fraction):
                if budget > 0:
                    # 筛选该小时时段的数据
                    hour_datas = test_data[test_data[:, hour_index] == t]
                    # 如果是初始状态，使用init_test_state
                    # 如果不是，就用tmp_test_state占位，即next_state
                    state = torch.tensor(init_test_state).float() if not t else torch.tensor(tmp_test_state).float()

                    action = rl_model.choose_best_action(state.unsqueeze(0))[0, 0].item()
                    if args.cont_tune:
                        curr_ratio = prev_ratio * (1 + action)
                    else:
                        curr_ratio = 1 + action
                    current_day_test_action[t] = (action, curr_ratio)
                    bid_datas = generate_bid_price((hour_datas[:, ctr_index] * hb_base / avg_ctr) / curr_ratio)
                    res_ = bid_main(bid_datas, hour_datas, budget)

                    # win_clks, real_clks, bids, imps, cost
                    # 数据是分天记录的
                    test_records = [test_records[i] + res_[i] for i in range(len(test_records))]
                    curr_rewards[0] += res_[0]
                    curr_rewards[1] += res_[2]
                    curr_rewards[2] += res_[-1]
                    budget -= res_[-1]

                    left_hour_ratio = (time_fraction - t) / (time_fraction) if t <= (time_fraction - 1) else 0
                    #left_hour_ratio = (time_fraction - 1 - t) / (time_fraction - 1) if t <= (
                    #        time_fraction - 1) else 0
                    curr_cpc = res_[-1] / res_[0] if res_[0] > 0 else 3 * args.tCPC
                    accu_cpc = test_records[-1] / test_records[0] if test_records[0] > 0 else 3 * args.tCPC
                    # avg_budget_ratio, cost_ratio, ctr, win_rate
                    #next_state = [(budget / current_day_budget) / left_hour_ratio if left_hour_ratio else (
                    #        budget / current_day_budget),
                    #              (budget / current_day_budget),
                    #              left_hour_ratio,
                    #              curr_cpc / args.tCPC,
                    #              accu_cpc / args.tCPC,                                 
                    #              res_[6] / current_day_budget,
                    #              res_[0] / res_[5] if res_[5] > 0 else 1.0,
                    #              res_[0] / res_[1] if res_[1] > 0 else 1.0,
                    #              res_[5] / res_[4] if res_[4] > 0 else 1.0,
                    #              test_records[0] / test_records[5] if test_records[5] > 0 else 1.0,
                    #              test_records[0] / test_records[1] if test_records[1] > 0 else 1.0,
                    #              test_records[5] / test_records[4] if test_records[4] > 0 else 1.0]
                    next_state = [(budget / current_day_budget) / left_hour_ratio if left_hour_ratio else (
                            budget / current_day_budget),
                                  res_[6] / current_day_budget,
                                  res_[0] / res_[5] if res_[5] else 0,
                                  res_[5] / res_[4] if res_[4] else 0,
                                  prev_ratio]
                    prev_ratio = curr_ratio
                    tmp_test_state = next_state

                    hour_t += 1
            curr_cpc = curr_rewards[2] / curr_rewards[0]
            if curr_rewards[0] > 0 and curr_cpc <= args.tCPC * (1 + args.margin):
                test_rewards = [test_rewards[i] + curr_rewards[i] for i in range(len(curr_rewards))]
            day_cpc += [curr_cpc]
            test_actions.append(current_day_test_action)

        ep_test_records.append([ep] + test_records + test_rewards)
        ep_test_actions.append(test_actions)
        print(ep, 'test', test_records, test_rewards, day_cpc, day_budget)

    train_record_df = pd.DataFrame(data=ep_train_records,
                                   columns=['ep', 'clks', 'real_clks', 'pctrs', 'real_pctrs', 'bids', 'imps', 'cost',
                                            'loss', 'rewards', 'constr_cost'])
    train_record_df.to_csv(
        os.path.join(result_path, 'uscb_train_records_' + str(args.tCPC * (1 + args.margin)) + '_' + str(
            args.budget_para) + '_' + str(args.lamda) + '_' + str(args.beta)  + '_' + str(args.reward_type) + '_' + str(args.cont_tune) + '.csv'), index=None)

    test_record_df = pd.DataFrame(data=ep_test_records,
                                  columns=['ep', 'clks', 'real_clks', 'pctrs', 'real_pctrs', 'bids', 'imps', 'cost',
                                           'clks_2', 'pctrs_2', 'cost_2'])
    test_record_df.to_csv(
        os.path.join(result_path, 'uscb_test_records_' + str(args.tCPC * (1 + args.margin)) + '_' + str(
            args.budget_para) + '_' + str(args.lamda) + '_' + str(args.beta) + '_' + str(args.reward_type) + '_' + str(args.cont_tune) + '.csv'), index=None)

    test_action_df = pd.DataFrame(data=ep_test_actions)
    test_action_df.to_csv(
        os.path.join(result_path, 'uscb_test_actions_' + str(args.tCPC * (1 + args.margin)) + '_' + str(
            args.budget_para) + '_' + str(args.lamda) + '_' + str(args.beta) + '_' + str(args.reward_type) + '_' + str(args.cont_tune) + '.csv'), index=None)
