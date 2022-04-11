# VPRobust

HT: Hood-Taylor element with P2-P1 (velocity-pressure).

BP: Bercovier-Pironneau P1isoP2-P1 (velocity-pressure).

BP+: Discontinuous Bercovier-Pironneau P1isoP2-(P1+P0) (velocity-pressure).

MINI: (P1+B3)-P1 (velocity-pressure).



| Test Name        | Description                                           | Elements   |
| ---------------- | ----------------------------------------------------- | ---------- |
| ArchimedesKitten | no flow example                                       | HT/BP/MINI |
| GaussEland       | large velocity w.r.t. pressure                        | HT/BP/MINI |
| CantorChameleon  | discrete div. w.r.t. increasing velocity              | HT/BP/MINI |
| WeilBull         | ArchimedesKitten with grad-div stabil. fixed mesh     | HT/BP/MINI |
| NashWolf         | GaussEland with grad-div stabil. fixed mesh           | HT/BP/MINI |
| BolyaiBadger     | CantorChameleon with grad-div stabil. fixed mesh      | HT/BP/MINI |

