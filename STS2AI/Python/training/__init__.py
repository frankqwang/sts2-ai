"""Training and evaluation infrastructure — extracted from train_hybrid.py and evaluate_ai.py.

Submodules:
  combat_ppo.py          — CombatRolloutBuffer, CombatPPOTrainer, mcts_train_step
  combat_diagnostics.py  — Combat trace/diagnostic logging functions
  game_decisions.py      — Map routing, card reward, shop decision heuristics
  eval_action_selection.py — NN/teacher/MCTS action selection strategies
  eval_game_state.py     — Game state tracking, loop detection, auto-progress
"""
