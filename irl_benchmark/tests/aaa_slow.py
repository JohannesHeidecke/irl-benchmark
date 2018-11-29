# def test_maze_model():
#     env = make_wrapped_env('MazeWorld0-v0', with_model_wrapper=True)
#
#     transitions, rewards = env._get_model_arrays()
#
#     # assert probability sums to 1.0
#     for s in range(transitions.shape[0]):
#         for a in range(transitions.shape[1]):
#             assert transitions[s, a].sum() == 1.0
#
#     assert isinstance(transitions, sparse.COO)
#     assert transitions.shape == (10241, 10, 10241)
#
#     assert rewards.shape == (10241, 10)