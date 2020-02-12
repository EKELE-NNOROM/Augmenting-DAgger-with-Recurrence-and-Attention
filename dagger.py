import train_policy
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
    policy = train_policy.main(args)
    cumulative_rewards = []

    for i in range(1,args.dagger_iterations):
        args.save_expert_actions = True
        args.expert_drives = False
        args.run_id = i
        args.timesteps= 100000
        # args.out_dir = "./dataset/train_full_6"
        args.out_dir = "./dataset/train"
        print ('GETTING EXPERT DEMONSTRATIONS')
        cumulative_rewards.append(racer.run(policy, args))
        print ('RETRAINING LEARNER ON AGGREGATED DATASET')
        args.weights_out_file = "./weights/learner_{}_dagger.weights".format(i)
        policy = train_policy.main(args,policy)

    cumulative_rewards.append(racer.run(policy, args))
    print(cumulative_rewards)
    with open('dagger_cum_rewards_nvidia.txt', 'w') as cum_rewards:
        for rewards in cumulative_rewards:
            cum_rewards.write(str(s) + '\n')

    plt.plot(figsize=(8, 4))


    	# plot the cumulative histogram

    plt.plot(cumulative_rewards)
    plt.grid(True)
    plt.legend(loc='right')
    plt.title('Cumulative Reward histograms')
    plt.xlabel('Dagger iterations')
    plt.ylabel('Rewards')


    plt.show()





    #
