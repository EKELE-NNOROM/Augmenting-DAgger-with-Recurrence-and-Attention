import train_policy_dagger
import racer
import argparse
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--dagger_iterations", help="", default=10)
    parser.add_argument("--freeze_last_n_layers", help="", default=0)
    parser.add_argument("--policy", type=str, default='OriginalDrivingPolicy',
                        help="driving policy of choice e.g. OriginalDrivingPolicy, RecurrentNetwork, AttentionNetwork or RecurrentAttention")
    #parser.add_argument("--output_log", type=str2bool, help="whether to output network logs", default=False)
    args = parser.parse_args()

    args.save_expert_actions = True
    args.expert_drives = True
    args.run_id = 0
    args.timesteps= 100000
    # args.out_dir = "./dataset/train_full_6"
    # args.train_dir =  "./dataset/train_full_6"
    # I made this change
    args.out_dir = "./dataset/train"
    args.train_dir =  "./dataset/train"
    racer.run(None, args)

    #print ('TRAINING LEARNER ON INITIAL DATASET')

    args.weighted_loss = True
    args.weights_out_file = "./weights/learner_0_full_6.weights"
    policy, train_loss, test_accuracy = train_policy_dagger.main(args)
    cumulative_rewards = []

    for i in range(1,args.dagger_iterations):
        # train_log = ''
        # test_log = ''
        # Added weighted loss because of the Attention Network
        args.weighted_loss = True
        args.save_expert_actions = True
        args.expert_drives = False
        args.run_id = i
        args.timesteps= 100000
        # args.out_dir = "./dataset/train_full_6"
        args.out_dir = "./dataset/train"
        print ('GETTING EXPERT DEMONSTRATIONS')
        cumulative_rewards.append(racer.run(policy, args))
        print ('RETRAINING LEARNER ON AGGREGATED DATASET')
        args.weights_out_file = "./weights/learner_{}_dagger_{}.weights".format(i, args.policy)

        train_log = './logs/{}_Loss_{}.txt'.format(args.policy, i)
        test_log = './logs/{}_Acc_{}.txt'.format(args.policy, i)
        policy, train_loss, test_accuracy = train_policy_dagger.main(args,policy)

        with open(train_log, 'w') as f_train:
            for loss in train_loss:
                f_train.write(str(loss) + '\n')
        with open(test_log, 'w') as f_test:
            for accuracy in test_accuracy:
                f_test.write(str(accuracy) + '\n')

    cumulative_rewards.append(racer.run(policy, args))
    #print(cumulative_rewards)
    cumulative_rewards_file = './logs/{}_Cum_Rewards.txt'.format(args.policy)
    with open(cumulative_rewards_file, 'w') as cum_rewards:
        for rewards in cumulative_rewards:
            cum_rewards.write(str(rewards) + '\n')







    #
