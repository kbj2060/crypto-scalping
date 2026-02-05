"""
MacroHFT 2단계: 국면별 전문가(Specialist) 학습
- trend / volatility / sideways 각각 해당 인덱스만 사용하여 학습
- 결과: data/agent_trend_best.pth, data/agent_volatility_best.pth, data/agent_sideways_best.pth
"""
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import config
from macroHFT.train_ppo import PPOTrainer


def train_specialist(regime_name, num_episodes=2000):
    if regime_name == "trend":
        target_key = "indices_trend"
    elif regime_name == "volatility":
        target_key = "indices_vol"
    elif regime_name == "sideways":
        target_key = "indices_chop"
    else:
        raise ValueError("regime_name must be one of: trend, volatility, sideways")

    trainer = PPOTrainer(enable_visualization=False)
    trainer.env.precompute_data()

    target_indices = getattr(trainer, target_key, trainer.all_indices)
    if not target_indices:
        print(f"⚠️ No indices for regime '{regime_name}', using all_indices.")
        target_indices = trainer.all_indices

    trainer.all_indices = target_indices
    trainer.trend_indices = target_indices

    base_path = f"data/agent_{regime_name}"
    config.AI_MODEL_PATH = f"{base_path}.pth"

    print(f"🚀 Training Specialist: {regime_name.upper()} Agent ({len(target_indices)} steps)")
    trainer.train(num_episodes=num_episodes)
    print(f"✅ Done. Check {base_path}_best.pth, {base_path}_last.pth")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MacroHFT specialist agents")
    parser.add_argument("--regime", type=str, choices=["trend", "volatility", "sideways"], required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args()
    train_specialist(args.regime, num_episodes=args.episodes)
