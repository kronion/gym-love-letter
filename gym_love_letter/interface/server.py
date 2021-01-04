import webbrowser

from flask import Flask
from stable_baselines3.ppo import PPO


from gym_love_letter.agents import HumanAgent, RandomAgent
from gym_love_letter.envs.base import LoveLetterMultiAgentEnv
from gym_love_letter.interface.api import make_api


app = Flask(__name__, static_url_path="/static", static_folder="dist")


def make_agents(env):
    human = HumanAgent()
    # load_path = "zoo/ppo_reward_bugfix4/latest/best_model"
    # load_path = "zoo/ppo_logging/2020-12-27T15:51:49/final_model"
    load_path = "zoo/ppo_kl/2020-12-27T16:28:42/final_model"
    model = PPO.load(load_path, env)
    # random1 = RandomAgent(env)
    # random2 = RandomAgent(env)

    return [human, model]  # random1, random2]


env = LoveLetterMultiAgentEnv(num_players=2, make_agents_cb=make_agents)


@app.route('/')
def index():
    return app.send_static_file("index.html")


api = make_api(env)
app.register_blueprint(api, url_prefix="/api")

# Launch the game
webbrowser.open("http://localhost:5000")
