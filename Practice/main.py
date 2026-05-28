import vmas
import hydra
from BenchMARL.benchmarl.experiment import Experiment
from BenchMARL.benchmarl.environments.vmas.common import VmasTask
from my_transport import MyTransportScenario

import pkgutil

def find_registry_path():
    # vmasパッケージ内のすべてのサブモジュールをスキャン
    for info in pkgutil.walk_packages(vmas.__path__, vmas.__name__ + "."):
        try:
            # モジュールをロード
            module = __import__(info.name, fromlist=["scenario_registry"])
            # そのモジュールの中に scenario_registry があるか確認
            if hasattr(module, "scenario_registry"):
                print(f"\n[発見!] 正しいインポートパスはこちらです:")
                print(f"from {info.name} import scenario_registry")
                return
        except Exception:
            continue
    print("\nscenario_registry が見つかりませんでした。")

find_registry_path()

# 1. VMAS側にシナリオを登録（物理エンジンへの登録）
vmas.simulator.scenario.scenario_registry["my_transport_scenario"] = MyTransportScenario

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    # 2. 実験を実行
    experiment = Experiment.from_config(cfg)
    experiment.run()

if __name__ == "__main__":
    main()