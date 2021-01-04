Interface
- [ ] Figure out how to manage the growing number of cards on the board
- [ ] Specify if the game is over, who won, and how to restart
    - [ ] Show cards of remaining players if the game ends with a reveal
    - [x] Show a win counter
- [ ] Don't allow player to see a game where they never got a chance to make a move?
- [ ] Create tooling to set an arbitrary state for the board
- [ ] Show training in interface, via webhooks?
- [ ] Allow player to choose number of opponents
- [ ] Add hotkeys to move faster

ML
- [ ] Export trained model to the front-end?
    - Seems like I can't do that until some bugs in ONNX exporting have been fixed.
        - Report a bug
- [-] Implement action masking in algorithm
    - [ ] Move the use_masking attribute?
    - [ ] Fork and make a PR
- [x] Experiment with different reward functions
    - [x] Extra reward for eliminating players?
    - [x] Extra reward for getting to the end of the game?
    - [x] Extra reward for ending game faster?
- [ ] GA tournament
- [ ] ELO scoring?
- [ ] Seems like agents can get stuck trying to play invalid actions, even after lots of training
    - Not with action masking applied, though
- [ ] Play the models against me?
    - Seems like the latest "best" model makes nonsensical moves 1:1. This is suspicious because the model I trained "heads-up" didn't seem to make these kinds of mistakes, but it fared worse in testing.

Engine
- [x] Don't make observation space dependant on the number of players
- [x] Decide if the empty action should ever be included in env.valid_actions
        - It shouldn't
