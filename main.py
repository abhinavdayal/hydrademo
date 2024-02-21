import hydra
from omegaconf import DictConfig, OmegaConf
import logging

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    X_train, y_train, X_test, y_test = hydra.utils.instantiate(cfg.dataloader)
    model = hydra.utils.instantiate(cfg.model)
    model.train(X_train, y_train)
    y_pred = hydra.utils.instantiate(cfg.evaluate, model, X_test)
    loss = hydra.utils.instantiate(cfg.metrics, y_test, y_pred)
    output = hydra.utils.instantiate(cfg.output, loss)
if __name__ == "__main__":
    my_app()