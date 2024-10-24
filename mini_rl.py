import os
import sys

def main(args):
    # system parameters:
    # args.ngpu_per_node
    # args.nnode_actor
    # args.nnode_learner
    
    # status_rank : args.rank
    resource_manager = ResourceManager(args.rank, args.ngpu_per_node, args.nnode_actor, args.nnode_learner)

    # vllm or megatron-lm. model initialization 
    if resource_manager.is_actor_node():
        actor_model = MegatronModel(args.init_checkpoint, args.rank, args.nnode_actor, args.ngpu_per_node, args.actor_model_config)

    # learner model initialization.
    if resource_manager.is_learner_node():
        learner_model = MegatronModel(args.init_checkpoint, args.rank, args.nnode_learner, args.ngpu_per_node, args.learner_model_config)

    env = DataEnv(args.env_config)
    if resource_manager.is_actor_node():
        env.init_actor_mode() #= DataEnv(args.env_config)
        play(env, actor_model, args)

    if resource_manager.is_learned_node():
        #replay_buffer = ReplayBuffer(args)
        env.init_learner_mode() #= DataEnv(args.env_config)
        learn(env, learner_model, args)

def learn(env, learner, args):
    batch_num = 0
    for i in range(0, args.learner_iteration):
        # rpc.rempte wait for the replay buffer from actor model.
        for batch in env.sample_replaybuffer(args.batch_size):
            learner.train(batch)
            batch_num += 1
            if batch_num % args.sync_freq == 0:
                env.push_to_modelbuffer(learner.model_weight)
                

def play(env, actor, args):
    for i in range(0, args.actor_iteration):
        for prompt in env:
            # multi-turn schedule.
            for r in range(0, env.round_schedule(i)):
                response = actor.predict(prompt)
                feedback = env.reward(response, prompt)
                # reflection + feedback.
                prompt = manu_prompt(prompt, response, feedback)
            # rpc.remote call.
            env.push_to_replaybuffer(prompt)
            
            if env.is_new_model():
                actor.sync(env.pull_modelbuffer())
                print('pull model weight.............')


if __name__ == "__main__":
    main(args)