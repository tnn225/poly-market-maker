from enum import Enum

Collateral = "Collateral"


class MyToken(Enum):
    A = "TokenA"
    B = "TokenB"

    def complement(self):
        return MyToken.B if self == MyToken.A else MyToken.A
