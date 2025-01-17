from functools import partial
import multiprocessing
import time
from datetime import datetime

from tqdm import tqdm

import numpy as np

import phyre

import ImgToObj


def chunkify(lst, n):
  return [lst[i::n] for i in range(n)]


def rect_intersect(rect1, rect2):
  l1 = rect1[0]
  r1 = rect1[2]
  l2 = rect2[0]
  r2 = rect2[2]
  # If one rectangle is on left side of other
  if(l1[0] > r2[0] or l2[0] > r1[0]):
    return False

  # If one rectangle is above other
  if(l1[1] < r2[1] or l2[1] < r1[1]):
    return False

  return True


def count_good_actions(task_ids, tier):
  cache = phyre.get_default_100k_cache(tier)
  task_data_dict = phyre.loader.load_compiled_task_dict()
  simulator = phyre.initialize_simulator(task_ids, tier)
  results = []
  stride = 5
  empty_action = phyre.simulator.scene_if.UserInput()
  max_actions = 100
  alpha = 1.0
  N = 5
  eval_stride = 1
  goal = 3.0 * 60.0/eval_stride

  for task_index in tqdm(range(len(task_ids)), desc='Evaluate tasks'):
    task_id = task_ids[task_index]
    task_data = task_data_dict[task_id]
    statuses = cache.load_simulation_states(task_id)
    _, _, images, _ = phyre.simulator.magic_ponies(
        task_data, empty_action, need_images=True, stride=stride)

    seq_data = ImgToObj.getObjectAndGoalSequence(images)

    discrete_actions = cache.action_array.tolist()
    good_action_count = 0
    solved_action_count = 0

    goal_type = ImgToObj.Layer.dynamic_goal.value
    if goal_type not in images[0]:
      goal_type = ImgToObj.Layer.static_goal.value

    tested_actions_count = 0
    tested_actions = np.array([[-1,-1,-1,1,0]])
    max_score = 0
    while tested_actions_count < max_actions and solved_action_count <= 0 and max_score < 1.0:
      random_action = np.random.random_sample((1,5))

      test_action_dist = np.linalg.norm(tested_actions[:,0:3] - random_action[:,0:3],axis=1)
      if np.any(test_action_dist <= tested_actions[:,3]) and np.random.random_sample() >= .25:
        continue

      if ImgToObj.check_seq_action_intersect(images[0],seq_data, stride, goal_type,np.squeeze(random_action[0:3])):
        sim_result = simulator.simulate_action(task_index, np.squeeze(random_action[:,0:3]), need_images=True, stride=eval_stride)
        if not sim_result.status.is_invalid():
          result_seq_data = ImgToObj.getObjectAndGoalSequence(sim_result.images)
          good_action_count += 1
          tested_actions_count += 1
          eval_dist = .05
          random_action[0,3] = eval_dist
          random_action[0,4] = 1.0 - np.linalg.norm(seq_data['object'][-1]['centroid'] - seq_data['goal'][-1]['centroid']) / 256.0
          random_action[0,4] += ImgToObj.objectTouchGoalSequence(sim_result.images) / goal
          if random_action[0,4] > max_score:
            max_score = random_action[0,4]
          tested_actions = np.concatenate((tested_actions,random_action),0)
          solved_task = sim_result.status.is_solved()
          solved_action_count += solved_task


    if solved_action_count <= 0:
      tested_actions = np.delete(tested_actions,0,0)
      theta = tested_actions[np.argmax(tested_actions[:,4]),0:3]
      theta_score = tested_actions[np.argmax(tested_actions[:,4]),4]
      b = 3
      mu = np.zeros(3)
      esp = np.eye(3)
      while tested_actions_count + 2*N+1 < max_actions and solved_action_count <= 0:
        old_theta = np.copy(theta)
        scores = np.zeros((N,2))
        deltas = np.zeros((N,3))
        i = 0
        while i < N and tested_actions_count + 2*N+1 < max_actions:
          delta = np.random.normal(0,.2 ,(1,3))      
          test_action_pos = theta + delta
          test_action_neg = theta - delta

          pos_score = 0
          sim_result_pos = simulator.simulate_action(task_index, np.squeeze(test_action_pos), need_images=True, stride=eval_stride)
          if not sim_result_pos.status.is_invalid():
            tested_actions_count += 1
            good_action_count += 1
            pos_result_seq_data = ImgToObj.getObjectAndGoalSequence(sim_result_pos.images)
            pos_score = 1.0 - np.linalg.norm(pos_result_seq_data['object'][-1]['centroid'] - pos_result_seq_data['goal'][-1]['centroid']) / 256.0
            pos_score += ImgToObj.objectTouchGoalSequence(sim_result_pos.images) / goal
            solved_task = sim_result_pos.status.is_solved()
            solved_action_count += solved_task
            
          neg_score = 0
          sim_result_neg = simulator.simulate_action(task_index, np.squeeze(test_action_neg), need_images=True, stride=eval_stride)
          if not sim_result_neg.status.is_invalid():
            tested_actions_count += 1
            good_action_count += 1
            neg_result_seq_data = ImgToObj.getObjectAndGoalSequence(sim_result_neg.images)
            neg_score = 1.0 - np.linalg.norm(neg_result_seq_data['object'][-1]['centroid'] - neg_result_seq_data['goal'][-1]['centroid']) / 256.0
            neg_score += ImgToObj.objectTouchGoalSequence(sim_result_neg.images) / goal
            solved_task = sim_result_neg.status.is_solved()
            solved_action_count += solved_task
          
          if pos_score != 0 or neg_score != 0:
            deltas[i,:] = delta
            scores[i,0] = pos_score
            scores[i,1] = neg_score
            i += 1

        max_scores = np.amax(scores,axis=1)
        max_index = np.argpartition(max_scores, b)[-b:]

        for i in max_index:
          if np.std(scores) == 0:
            print(task_id)
            theta = theta + (alpha / (b)) * (scores[i,0] - scores[i,1]) * deltas[i,:]
          else:
            theta = theta + (alpha / (b*np.std(scores[max_index]))) * (scores[i,0] - scores[i,1]) * deltas[i,:]
        
        sim_result = simulator.simulate_action(task_index, np.squeeze(theta), need_images=True, stride=eval_stride)
        
        if not sim_result.status.is_invalid():
          result_seq_data = ImgToObj.getObjectAndGoalSequence(sim_result.images)
          score = 1.0 - np.linalg.norm(result_seq_data['object'][-1]['centroid'] - result_seq_data['goal'][-1]['centroid']) / 256.0
          score += ImgToObj.objectTouchGoalSequence(sim_result.images) / goal
          #print(task_id,theta,score,old_theta,theta_score,sim_result.status.is_solved(),tested_actions_count)
          good_action_count += 1
          tested_actions_count += 1
          solved_task = sim_result.status.is_solved()
          solved_action_count += solved_task
        else:
          theta = old_theta

    results.append({'num_good': good_action_count,
                    'num_solved': solved_action_count, 'num_total': len(discrete_actions)})

  return results


tier = 'ball'
cache = phyre.get_default_100k_cache(tier)
task_ids = list(cache.task_ids)
task_ids.sort()
task_ids = task_ids
simulator = phyre.initialize_simulator(task_ids, tier)

print(len(task_ids))

pool_count = 4
pool = multiprocessing.Pool(pool_count)
partial_worker = partial(
    count_good_actions,
    tier=tier)

t0 = time.time()
results_list = pool.imap(partial_worker, chunkify(task_ids, pool_count))
total_actions = []
reduction_count = []
percent_solved = []
no_good_count = 0
no_sol_count = 0

for results in results_list:
  for result in results:
    total_actions.append(result['num_total'])
    if result['num_good'] != 0:
      reduction_count.append(result['num_good'])
      percent_solved.append(result['num_solved'] / result['num_good'])
    else:
      no_good_count += 1
    if result['num_solved'] == 0:
      no_sol_count += 1
t1 = time.time()
print((t1-t0)/len(task_ids), "Avg Time")

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H%M%S")

f = open("./stats/random_agent_with_seqARS_stats{}.txt".format(dt_string), "w+")

print("Total Actions Mean:", np.mean(total_actions), file=f)
print("Reduction Mean:", np.mean(reduction_count),
      "STD:", np.std(reduction_count), file=f)
print("Reduction Max:", np.max(reduction_count), "Reduction Min:", np.min(reduction_count), file=f)
print("Percent Solved Mean:", np.mean(percent_solved),
      "STD:", np.std(percent_solved), file=f)
print("Percent With No Good action:", no_good_count/len(task_ids), file=f)
print("Percent With No solution:", no_sol_count/len(task_ids), file=f)
