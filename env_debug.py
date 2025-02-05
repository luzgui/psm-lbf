import time

def _game_loop(environment, render, tester, pol_func):
    """
    Runs a single episode of the ForagingEnv_r environment.
    """
    obs = environment.reset()  # Reset the environment
    done = False
    iteration = 0

    if render:
        environment.render()
        time.sleep(0.01)

    while not done:
        print(f'Iteration: {iteration}')

        # Generate random actions if no policy tester is provided
        if not tester:
            actions = {agent_id: environment.action_space.sample() for agent_id in environment.agents_id}
            #import pdb
            #pdb.set_trace()
        else:
            actions = {
                aid: tester.compute_single_action(obs[aid], policy_id=pol_func(aid))
                for aid in environment.agents_id
            }

        print('Actions:', actions)

        # Take a step in the environment
        obs, nreward, ndone, _ = environment.step(actions)

        # Print rewards if any agent received a reward
        if sum(nreward.values()) > 0:
            print('Reward:', nreward)

        if render:
            environment.render()
            time.sleep(0.6)

        # Check if the episode is done
        done = ndone['__all__']
        iteration += 1

def main_loop(game_count, environment, tester, pol_func):
    """
    Runs multiple episodes of the environment for debugging.
    """
    render = True
    for episode in range(game_count):
        print(f"\n--- Episode {episode + 1} ---")
        _game_loop(environment, render, tester, pol_func)
