from flask import jsonify
import hydra
from omegaconf import OmegaConf

from server import app, EvoConfig, exp_config


evo_config = None


@app.route('/get_evo_args', methods=['GET'])
def get_evo_args():
    global exp_config
    exp_config = evo_config
    return jsonify(OmegaConf.to_container(evo_config))


@hydra.main(config_name="evo", version_base="1.3")
def main(cfg: EvoConfig):
    global evo_config, exp_config
    evo_config, exp_config = cfg, cfg
    app.run(port=cfg.port)


if __name__ == '__main__':
    main()