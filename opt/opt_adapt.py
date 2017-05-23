import hyperopt.hp
import opt.hyper

space_vals = {
    #    'conv4_3_cnt': hyperopt.hp.quniform('conv4_3_cnt', 0, 512, 1),
    #    'conv5_3_cnt': hyperopt.hp.quniform('conv5_3_cnt', 0, 512, 1),
    #    'sigma_train': hyperopt.hp.uniform('sigma_train', 0.0, 2.0),
    #    'sigma_update': hyperopt.hp.uniform('sigma_update', 0.0, 2.0),
    'update_initial_factor': hyperopt.hp.uniform('update_initial_factor', 0.0, 1.0),
    'update_current_factor': hyperopt.hp.uniform('update_current_factor', 0.0, 1.0),
    #    'update_threshold': hyperopt.hp.uniform('update_threshold', 0.0, 1.0),
    #    'update_use_quality': hyperopt.hp.choice('update_use_quality', ['true', 'false']),
    #    'particle_count': hyperopt.hp.quniform('particle_count', 100, 2000, 100),
    #
    #'cons_learning_rate': hyperopt.hp.choice('cons_learning_rate', [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]),
    #'cons_max_iterations': hyperopt.hp.quniform('cons_max_iterations', 10, 100, 10),
    #'cons_min_cost': hyperopt.hp.choice('cons_min_cost', ['null', 0.001, 0.01, 0.03, 0.06, 0.1, 0.2, 0.5]),
    #
    #'cons_conv1_channels': hyperopt.hp.choice('cons_conv1_channels', [16, 32, 64, 72]),
    #'cons_conv1_kernel_size': hyperopt.hp.choice('cons_conv1_kernel_size', [3, 5, 7, 9, 11, 13, 15]),
    #'cons_conv2_kernel_size': hyperopt.hp.choice('cons_conv2_kernel_size', [3, 5, 7, 9, 11, 13, 15]),
}

template = """
# tracker.yaml
# defines how the tracker is set up and what samples will be tracked
# Can be fed to program via command line argument --tracker [path]

# OPT: opt_adapt
# find out, what update factors are good

sroi_size: [368, 368]
#sroi_size: [496, 496]
mask_size: null
extractor_net: vgg16
features:
  - [conv4_3, 384]
  - [conv5_3, 384]
random_seed: null
selector:
  max_iterations: 50
  min_cost: 0.01
  net:
    name: selector_net
    cost: mean_square
    optimizer: [adam, {learning_rate: 0.000001}]
    layers:
      - type: dropout
        name: dropout
        keep_prob: 0.7
      - type: conv
        name: conv1
        weight_initial: [truncated_normal, {stddev: 0.0000001}]
        bias_initial: zeros 
        kernel_size: 3
        channels: 1
      
consolidator:
  max_iterations: 100
  min_cost: null
  sigma_train: 1.0
  sigma_update: 1.0
  update_threshold: 0.2
  update_initial_factor: ${update_initial_factor}
  update_use_quality: True
  update_current_factor: ${update_current_factor}
  net:
    name: consolidator_net
    cost: mean_square
    optimizer: [adam, {learning_rate: 0.00001}]
    layers:
      - type: conv
        name: conv1
        weight_initial: [truncated_normal, {stddev: 0.0000001}]
        bias_initial: 0.1
        kernel_size: 9
        channels: 32
      - type: conv
        name: conv2
        weight_initial: [truncated_normal, {stddev: 0.0000001}]
        bias_initial: 0
        kernel_size: 5
        channels: 1

pursuer:
  particle_count: 1000
  target_lower_limit: 0.1
  target_punish_low: -0.1
  target_punish_outside: 0.0

# = tracking = 
# which samples shall be tracked? Accepts samples, data sets, and data collections
# examples: ['tb100/MotorRolling', 'SET/tb100', 'COLLECTION/tb100_probe']
tracking:
    - COLLECTION/tb100_adapt
"""

opt.hyper.run(
    space_vals=space_vals,
    max_evals=1,
    template=template,
)
