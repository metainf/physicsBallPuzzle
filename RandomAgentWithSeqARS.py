import random
import faulthandler
from functools import partial
import multiprocessing
from datetime import datetime

from scipy.stats import gaussian_kde

import numpy as np

from tqdm import tqdm

import phyre

import ImgToObj

def evaluate_agent(task_ids, tier,solved_actions_pdf):
  cache = phyre.get_default_100k_cache(tier)
  evaluator = phyre.Evaluator(task_ids)
  simulator = phyre.initialize_simulator(task_ids, tier)
  task_data_dict = phyre.loader.load_compiled_task_dict()
  stride = 50
  eval_stride = 5
  goal = 3.0 * 60.0/eval_stride
  empty_action = phyre.simulator.scene_if.UserInput()
  tasks_solved = 0
  alpha = 1.0
  N = 5
  b = 3
  max_actions = 100
  max_search_actions = phyre.MAX_TEST_ATTEMPTS - (N*2+1) * 5

  for task_index in tqdm(range(len(task_ids)), desc='Evaluate tasks'):
    task_id = task_ids[task_index]
    task_type = task_id.split(":")[0]
    task_data = task_data_dict[task_id]
    statuses = cache.load_simulation_states(task_id)
    _, _, images, _ = phyre.simulator.magic_ponies(
        task_data, empty_action, need_images=True, stride=stride)

    evaluator.maybe_log_attempt(task_index, phyre.simulation_cache.NOT_SOLVED)

    seq_data = ImgToObj.getObjectAndGoalSequence(images)

    goal_type = ImgToObj.Layer.dynamic_goal.value
    if goal_type not in images[0]:
      goal_type = ImgToObj.Layer.static_goal.value

    tested_actions = np.array([[-1,-1,-1,1,0]])

    solved_task = False
    max_score = 0
    while evaluator.get_attempts_for_task(task_index) < max_actions and not solved_task and max_score < 1.0:
      random_action = np.random.random_sample((1,5))
      if task_type in solved_actions_pdf and np.random.random_sample() >= .25:
        random_action[0,0:3] = np.squeeze(solved_actions_pdf[task_type].resample(size=1))
      
      test_action_dist = np.linalg.norm(tested_actions[:,0:3] - random_action[:,0:3],axis=1)
      if np.any(test_action_dist <= tested_actions[:,3]) and np.random.random_sample() >= .25:
        continue
      if ImgToObj.check_seq_action_intersect(images[0],seq_data, stride, goal_type,np.squeeze(random_action[0:3])):

        sim_result = simulator.simulate_action(task_index, np.squeeze(random_action[:,0:3]), need_images=True, stride=eval_stride)
        evaluator.maybe_log_attempt(task_index, sim_result.status)
        if not sim_result.status.is_invalid():
          score = ImgToObj.objectTouchGoalSequence(sim_result.images)
          eval_dist = .1
          random_action[0,3] = eval_dist
          random_action[0,4] = 1.0 - np.linalg.norm(seq_data['object'][-1]['centroid'] - seq_data['goal'][-1]['centroid']) / 256.0
          random_action[0,4] += ImgToObj.objectTouchGoalSequence(sim_result.images) / goal
          if random_action[0,4] > max_score:
            max_score = random_action[0,4]
          tested_actions = np.concatenate((tested_actions,random_action),0)
          solved_task = sim_result.status.is_solved()
          tasks_solved += solved_task
    
    if not solved_task and evaluator.get_attempts_for_task(task_index) < phyre.MAX_TEST_ATTEMPTS:
      tested_actions = np.delete(tested_actions,0,0)
      theta = tested_actions[np.argmax(tested_actions[:,4]),0:3]
      theta_score = tested_actions[np.argmax(tested_actions[:,4]),4]

      while evaluator.get_attempts_for_task(task_index) + 2*N+1 < phyre.MAX_TEST_ATTEMPTS and not solved_task:
        scores = np.zeros((N,2))
        deltas = np.zeros((N,3))
        i = 0
        old_theta = np.copy(theta)
        while i < N and evaluator.get_attempts_for_task(task_index) + 2*N+1 < max_actions:
          delta = np.random.normal(0,.2 ,(1,3))
          test_action_pos = theta + delta
          test_action_neg = theta - delta

          pos_score = 0
          sim_result_pos = simulator.simulate_action(task_index, np.squeeze(test_action_pos), need_images=True, stride=eval_stride)
          evaluator.maybe_log_attempt(task_index, sim_result_pos.status)
          if not sim_result_pos.status.is_invalid():
            pos_result_seq_data = ImgToObj.getObjectAndGoalSequence(sim_result_pos.images)
            pos_score = 1.0 - np.linalg.norm(pos_result_seq_data['object'][-1]['centroid'] - pos_result_seq_data['goal'][-1]['centroid']) / 256.0
            pos_score += ImgToObj.objectTouchGoalSequence(sim_result_pos.images) / goal
            solved_task = sim_result_pos.status.is_solved()
            tasks_solved += solved_task
            
          neg_score = 0
          sim_result_neg = simulator.simulate_action(task_index, np.squeeze(test_action_neg), need_images=True, stride=eval_stride)
          evaluator.maybe_log_attempt(task_index, sim_result_neg.status)
          if not sim_result_neg.status.is_invalid():
            neg_result_seq_data = ImgToObj.getObjectAndGoalSequence(sim_result_neg.images)
            neg_score = 1.0 - np.linalg.norm(neg_result_seq_data['object'][-1]['centroid'] - neg_result_seq_data['goal'][-1]['centroid']) / 256.0
            neg_score += ImgToObj.objectTouchGoalSequence(sim_result_neg.images) / goal
            solved_task = sim_result_neg.status.is_solved()
            tasks_solved += solved_task

          if pos_score != 0 or neg_score != 0:
            deltas[i,:] = delta
            scores[i,0] = pos_score
            scores[i,1] = neg_score
            i += 1

        max_scores = np.amax(scores,axis=1)
        max_index = np.argpartition(max_scores, b)[-b:]
        for i in max_index:
          if np.std(scores) <= 1e-6:
            print(task_id)
            theta = theta + (alpha / (b)) * (scores[i,0] - scores[i,1]) * deltas[i,:]
          else:
            theta = theta + (alpha / (b*np.std(scores[max_index]))) * (scores[i,0] - scores[i,1]) * deltas[i,:]
        
        sim_result = simulator.simulate_action(task_index, np.squeeze(theta), need_images=True, stride=eval_stride)
        evaluator.maybe_log_attempt(task_index, sim_result.status)
        if not sim_result.status.is_invalid():
          result_seq_data = ImgToObj.getObjectAndGoalSequence(sim_result.images)
          score = 1.0 - np.linalg.norm(result_seq_data['object'][-1]['centroid'] - result_seq_data['goal'][-1]['centroid']) / 256.0
          score += ImgToObj.objectTouchGoalSequence(sim_result.images) / goal
          solved_task = sim_result.status.is_solved()
          tasks_solved += solved_task
        else:
          theta = old_theta

  print(tasks_solved, "Tasks solved out of ", len(task_ids), "Total Tasks")
  return (evaluator.get_aucess(), tasks_solved,len(task_ids))

def train_kde(tasks,tier):
  cache = phyre.get_default_100k_cache(tier)
  all_solved_actions = {}
  for task_id in tasks:
    task_type = task_id.split(":")[0]
    statuses = cache.load_simulation_states(task_id)
    solved_actions = cache.action_array[statuses==phyre.simulation_cache.SOLVED,:]
    if task_type not in all_solved_actions:
      all_solved_actions[task_type] = solved_actions
    else:
      all_solved_actions[task_type] = np.concatenate((all_solved_actions[task_type],solved_actions),0)
  
  solved_actions_pdf = {}
  for task_type in all_solved_actions.keys():
    solved_actions_pdf[task_type] = gaussian_kde(np.transpose(all_solved_actions[task_type]),bw_method="silverman")

  return solved_actions_pdf

def worker(fold_id, eval_setup):
  train, dev, test = phyre.get_fold(eval_setup, fold_id)
  action_tier = phyre.eval_setup_to_action_tier(eval_setup)
  solved_actions_pdf = train_kde(train,action_tier)
  return evaluate_agent(test, action_tier,solved_actions_pdf)


faulthandler.enable()
random.seed(0)
np.random.seed(0)
#eval_setups = ['ball_cross_template', 'ball_within_template']
#eval_setups = ['ball_within_template']
eval_setups = ['ball_cross_template']
fold_ids = list(range(0, 10))
print('eval setups', eval_setups)
print('fold ids', fold_ids)

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H%M%S")

f = open("random_agent_with_seqARS_results{}.csv".format(dt_string), "w+")
print('eval_setup,fold_id,AUCESS,Solved,Total,Percent_Solved', file=f)

for eval_setup in eval_setups:
  pool = multiprocessing.Pool(8)
  partial_worker = partial(
      worker,
      eval_setup=eval_setup)
  results = pool.imap(partial_worker, list(range(0, 10)))
  for fold_id, result in enumerate(results):
    print('{},{},{},{},{},{}'.format(eval_setup, fold_id, result[0],result[1],result[2],float(result[1])/result[2]), file=f)
