from enum import Enum
from . import BabyAITextEnv, AlfWorldEnv

class EnvEnum(Enum):
    babyai_text = BabyAITextEnv
    alfworld = AlfWorldEnv