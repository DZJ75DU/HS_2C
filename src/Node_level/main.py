from core.pipeline import HS2CPipeline
from configs.config import Config


def main():
    config = Config().to_dict()

    pipeline = HS2CPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
