# Original Specification

Measure how an LLM adapts its choice behavior in a two-armed bandit task after a hidden, one-time reversal of the reward contingency. The model must track feedback across trials and update its policy without ever being told the underlying rule.

Two actions: A, B.

One trial = one action choice by the model, followed by a binary reward (0 or 1) reported back to it.

Contingency, phase 1 (trials 0 .. r-1): B -> 1, A -> 0.

Hidden reversal at trial r, drawn once per run.

Contingency, phase 2 (trials r .. n-1): A -> 1, B -> 0.

Reward is deterministic given the active contingency (no reward noise).

The model is never told r, never told which contingency is active, and never told the reversal occurred. It only sees its own choice and the resulting reward, each trial, for the whole run.

A "run" (a.k.a. "game") is one full sequence of n_trials trials with one independently drawn reversal point. Runs are independent: no state, RNG, or conversation history crosses runs. The model knows the number of trials and has a task to maximize its score. Everything should be stored in a database. Database's name should go like model_name_config_number_date_time.sql. The date and time (with microseconds) of the run goes as a seed by default. The model has to answer in a json format ONLY, without any additional reasonings. If any error occurs, including wrong answer's format, the run has to be stopped and recorded to the db. The error's description goes to the db too. Everything which is sent to and received from the mode and errors should be displayed on the screen.

Information which has to go to the config:

* provider  - Anthropic, OpenAI, Together.ai
* model name
* Temperature
* Max tokens
* Seed #optional
* N trials
* Reversal trial
* Choice A: <string> #Choices may be A and B or Left and Right or Red and Blue and so on.
* Choice B: <string>
* Initial contingency, for example: {"B": 1, "A": 0}
* Post reversal contingency, for example: {"B": 0, "A": 1}
* Reward size

You are welcome to propose more.

There should be a separate file with prompts. Knowing everything above, fill it with preliminary versions.

Config, all the prompts, all models answers go to the db. If you have any questions and suggestions on how to improve the work flow, please, tell me. Let's discuss them before implementation.
