from typing import Tuple
from torch import Tensor


class T:
    _BATCH = Tuple[Tensor, Tensor]


class Names:
    TRAINING = "Training"
    UNKNOWN_VALID = "Unkonwn Validation"
    KNOWN_VALID = "Known Validation"
